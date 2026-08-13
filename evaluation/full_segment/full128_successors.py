"""Exact cached evaluation and reporting for metadata-face-eligible successors."""

from __future__ import annotations

import hashlib
import math
import mmap
import os
import re
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Self

import numpy as np

from evaluation.full128_successor_reporting import (
    _PUBLIC_LIMITATIONS,
    DEV_SELECTION_SCHEMA,
    ENROLLMENT_KS,
    EVALUATION_SCOPES,
    PAIRED_BOOTSTRAP_SCHEMA,
    PUBLIC_REPORT_SCHEMA,
    Full128SuccessorEvaluationError,
    _reject_private_public_keys,
    _sanitize_paired_result,
    _sanitize_scope_aggregate,
    _sanitize_selection_receipt,
    validate_public_successor_evaluation_report,
)
from foundation.provenance import content_sha256
from embedding.methods.full_segment.face_visible import (
    AUTHORITATIVE_PANEL_SCHEMA,
    build_authoritative_face_visible_panel,
    validate_face_visible_successor_inventory_bundle,
    validate_score_blind_face_visible_panel,
)

CACHE_SCHEMA = "cvi.full128_successor_embedding_cache.v1"
EVALUATION_PANEL_SCHEMA = "cvi.full128_successor_evaluation_panel.v1"
AUTHORITATIVE_EVALUATION_PANEL_SCHEMA = "cvi.full128_successor_evaluation_panel.v2"
PRIVATE_REPORT_SCHEMA = "cvi.full128_successor_private_evaluation.v2"
LEGACY_PRIVATE_REPORT_SCHEMA = "cvi.full128_successor_private_evaluation.v1"
TERMINAL_DECISION_SCHEMA = "cvi.full128_successor_multiseed_terminal_decision.v1"
EMBEDDING_DIMENSION = 128
_VECTOR_BYTES = EMBEDDING_DIMENSION * 4
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_PACK_BYTES = 64 * 1024 * 1024 * 1024
_METRICS = ("Rank-1", "Rank-5", "Rank-10", "MRR")


@dataclass(frozen=True, slots=True)
class ValidatedSuccessorEmbeddingCache:
    """Immutable mapped view of a fully validated external packed cache."""

    descriptor: dict[str, Any]
    pack_path: Path
    _vectors: np.ndarray = dataclass_field(repr=False, compare=False)
    _mapping: mmap.mmap | None = dataclass_field(repr=False, compare=False)

    def load_embeddings(self, sample_tokens: Sequence[str]) -> np.ndarray:
        if self._mapping is None:
            raise Full128SuccessorEvaluationError("embedding cache is closed")
        tokens = tuple(sample_tokens)
        if len(tokens) != len(set(tokens)):
            raise Full128SuccessorEvaluationError("embedding request repeats a token")
        index = {
            token: row for row, token in enumerate(self.descriptor["sample_tokens"])
        }
        if unknown := set(tokens) - set(index):
            raise Full128SuccessorEvaluationError(
                f"embedding cache omits requested samples: {sorted(unknown)[:3]}"
            )
        if not tokens:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        return self._vectors[[index[token] for token in tokens]].copy()

    def close(self) -> None:
        """Release the owned mapping; repeated closure is harmless."""

        mapping = self._mapping
        if mapping is None:
            return
        empty = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        empty.setflags(write=False)
        object.__setattr__(self, "_vectors", empty)
        object.__setattr__(self, "_mapping", None)
        mapping.close()

    def __enter__(self) -> Self:
        if self._mapping is None:
            raise Full128SuccessorEvaluationError("embedding cache is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_score_blind_fixed_evaluation_panel(
    successor_inventory_bundle: object,
    source_panel: object | None = None,
    *,
    face_protocol_v2_bundle: object | None = None,
    gallery_query_panel_bundle: object | None = None,
) -> dict[str, Any]:
    """Bind materialization terminals to the metadata panel without panel shrinkage."""

    v2_requested = (
        face_protocol_v2_bundle is not None or gallery_query_panel_bundle is not None
    )
    if v2_requested:
        if (
            face_protocol_v2_bundle is None
            or gallery_query_panel_bundle is None
            or source_panel is not None
        ):
            raise Full128SuccessorEvaluationError(
                "governance v2 evaluation requires protocol and panel only"
            )
        source_panel = build_authoritative_face_visible_panel(
            face_protocol_v2_bundle, gallery_query_panel_bundle
        )
    if source_panel is None:
        raise Full128SuccessorEvaluationError(
            "evaluation requires a fixed panel authority"
        )
    inventory = validate_face_visible_successor_inventory_bundle(
        successor_inventory_bundle, verify_artifacts=False
    )
    panel = validate_score_blind_face_visible_panel(source_panel)
    binding = inventory["source_binding"]
    if binding["fixed_panel_sha256"] != panel["panel_sha256"]:
        raise Full128SuccessorEvaluationError(
            "successor inventory and source fixed panel differ"
        )
    successor_rows = {
        row["sample_token"]: row
        for row in inventory["inventory"]["successor_population"]
    }
    panel_tokens = {row["sample_token"] for row in panel["records"]}
    if panel_tokens != set(successor_rows):
        missing = sorted(panel_tokens - set(successor_rows))
        extra = sorted(set(successor_rows) - panel_tokens)
        raise Full128SuccessorEvaluationError(
            f"fixed panel and successor population differ; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    records: list[dict[str, Any]] = []
    authoritative = panel["schema_version"] == AUTHORITATIVE_PANEL_SCHEMA
    for source in panel["records"]:
        joined = successor_rows[source["sample_token"]]
        uses: dict[str, dict[str, str | None]] = {}
        for k in ENROLLMENT_KS:
            use = dict(source["uses_by_enrollment_k"][str(k)])
            if joined["state"] != "USABLE" and use["use"] in {"QUERY", "GALLERY"}:
                use = {
                    "use": "TERMINAL_EXCLUSION",
                    "reason": joined["terminal_reason"],
                }
            uses[str(k)] = use
        record = {
            "sample_token": source["sample_token"],
            "registered_identity_id": source["registered_identity_id"],
            "dataset_name": source["dataset_name"],
            "scope": source["scope"],
            "duplicate_component": source["duplicate_component"],
            "source_panel_record_sha256": source["record_sha256"],
            "successor_inventory_record_sha256": joined["record_sha256"],
            "uses_by_enrollment_k": uses,
        }
        if authoritative:
            record["authoritative_cohort_member"] = source[
                "authoritative_cohort_member"
            ]
        records.append(record)
    records.sort(key=lambda item: item["sample_token"])
    _reject_evaluation_panel_leakage(records)
    membership_contract = (
        "AUTHORITATIVE_SHARED_QUERY_NESTED_K" if authoritative else "LEGACY_V1"
    )
    cohorts = _panel_cohorts(records, membership_contract=membership_contract)
    required = sorted(
        {
            token
            for cohort in cohorts
            if cohort["status"] == "AVAILABLE"
            for token in (
                *cohort["query_sample_tokens"],
                *cohort["gallery_sample_tokens"],
            )
        }
    )
    payload = {
        "schema_version": (
            AUTHORITATIVE_EVALUATION_PANEL_SCHEMA
            if authoritative
            else EVALUATION_PANEL_SCHEMA
        ),
        "source_successor_inventory_bundle_sha256": inventory["bundle_sha256"],
        "source_successor_inventory_sha256": inventory["inventory_sha256"],
        "source_fixed_panel_sha256": panel["panel_sha256"],
        "score_inputs_used": False,
        "terminal_policy": "PRESERVE_AND_LABEL;NEVER_REPLACE_OR_SHRINK",
        "scope_labels": list(EVALUATION_SCOPES),
        "records": records,
        "cohorts": cohorts,
        "required_sample_tokens": required,
        "required_sample_tokens_sha256": content_sha256(required),
    }
    if authoritative:
        payload["membership_contract"] = membership_contract
    return {**payload, "panel_sha256": content_sha256(payload)}


def build_authoritative_fixed_evaluation_panel(
    successor_inventory_bundle: object,
    face_protocol_v2_bundle: object,
    gallery_query_panel_bundle: object,
) -> dict[str, Any]:
    """Consume governance artifacts directly and retain their fixed memberships."""

    source_panel = build_authoritative_face_visible_panel(
        face_protocol_v2_bundle, gallery_query_panel_bundle
    )
    return build_score_blind_fixed_evaluation_panel(
        successor_inventory_bundle, source_panel
    )


def validate_fixed_evaluation_panel(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Full128SuccessorEvaluationError(
            "successor evaluation panel fields differ"
        )
    authoritative = value.get("schema_version") == AUTHORITATIVE_EVALUATION_PANEL_SCHEMA
    expected = {
        "schema_version",
        "source_successor_inventory_bundle_sha256",
        "source_successor_inventory_sha256",
        "source_fixed_panel_sha256",
        "score_inputs_used",
        "terminal_policy",
        "scope_labels",
        "records",
        "cohorts",
        "required_sample_tokens",
        "required_sample_tokens_sha256",
        "panel_sha256",
    }
    if authoritative:
        expected.add("membership_contract")
    _keys(value, expected, "successor evaluation panel")
    panel = dict(value)
    payload = {key: item for key, item in panel.items() if key != "panel_sha256"}
    if (
        panel["schema_version"]
        not in {EVALUATION_PANEL_SCHEMA, AUTHORITATIVE_EVALUATION_PANEL_SCHEMA}
        or panel["panel_sha256"] != content_sha256(payload)
        or panel["score_inputs_used"] is not False
        or panel["scope_labels"] != list(EVALUATION_SCOPES)
        or panel["terminal_policy"] != "PRESERVE_AND_LABEL;NEVER_REPLACE_OR_SHRINK"
    ):
        raise Full128SuccessorEvaluationError(
            "successor evaluation panel contract differs"
        )
    for field in (
        "source_successor_inventory_bundle_sha256",
        "source_successor_inventory_sha256",
        "source_fixed_panel_sha256",
        "required_sample_tokens_sha256",
        "panel_sha256",
    ):
        _sha(panel[field], field)
    records = panel["records"]
    if not isinstance(records, list):
        raise Full128SuccessorEvaluationError(
            "evaluation panel records must be an array"
        )
    tokens = [row.get("sample_token") for row in records if isinstance(row, Mapping)]
    if len(tokens) != len(records) or tokens != sorted(set(tokens)):
        raise Full128SuccessorEvaluationError(
            "evaluation panel records must be uniquely sorted"
        )
    for row in records:
        _validate_evaluation_panel_record(row, authoritative=authoritative)
    _reject_evaluation_panel_leakage(records)
    membership_contract = panel["membership_contract"] if authoritative else "LEGACY_V1"
    if authoritative and membership_contract != "AUTHORITATIVE_SHARED_QUERY_NESTED_K":
        raise Full128SuccessorEvaluationError(
            "evaluation panel membership contract differs"
        )
    if panel["cohorts"] != _panel_cohorts(
        records, membership_contract=membership_contract
    ):
        raise Full128SuccessorEvaluationError("evaluation panel cohorts differ")
    required = sorted(
        {
            token
            for cohort in panel["cohorts"]
            if cohort["status"] == "AVAILABLE"
            for token in (
                *cohort["query_sample_tokens"],
                *cohort["gallery_sample_tokens"],
            )
        }
    )
    if panel["required_sample_tokens"] != required or panel[
        "required_sample_tokens_sha256"
    ] != content_sha256(required):
        raise Full128SuccessorEvaluationError(
            "evaluation panel required sample population differs"
        )
    return panel


def build_successor_embedding_cache_descriptor(
    *,
    successor_id: str,
    pack_path: Path,
    sample_tokens: Sequence[str],
    successor_inventory_bundle_sha256: str,
    successor_inventory_sha256: str,
    evaluation_panel_sha256: str,
    model_manifest_sha256: str,
    checkpoint_sha256: str,
    preprocessing_manifest_sha256: str,
    embedding_manifest_sha256: str,
) -> dict[str, Any]:
    """Describe an existing raw f32le pack after exact vector validation."""

    _nonempty(successor_id, "successor_id")
    tokens = list(sample_tokens)
    if not tokens or tokens != sorted(set(tokens)):
        raise Full128SuccessorEvaluationError(
            "cache sample tokens must be non-empty, unique, and sorted"
        )
    for token in tokens:
        _sha(token, "cache sample token")
    for value, label in (
        (successor_inventory_bundle_sha256, "successor inventory bundle"),
        (successor_inventory_sha256, "successor inventory"),
        (evaluation_panel_sha256, "evaluation panel"),
        (model_manifest_sha256, "model manifest"),
        (checkpoint_sha256, "checkpoint"),
        (preprocessing_manifest_sha256, "preprocessing manifest"),
        (embedding_manifest_sha256, "embedding manifest"),
    ):
        _sha(value, label)
    path = _regular_file(pack_path, "successor embedding pack")
    expected_size = len(tokens) * _VECTOR_BYTES
    if path.stat().st_size != expected_size:
        raise Full128SuccessorEvaluationError(
            "successor embedding pack byte size differs"
        )
    descriptor_fd, mapping, initial = _open_regular_mapping(
        path, expected_size=expected_size
    )
    matrix = np.ndarray((len(tokens), EMBEDDING_DIMENSION), dtype="<f4", buffer=mapping)
    try:
        _validate_matrix(matrix, len(tokens))
        pack_sha256 = _mapped_sha256(mapping)
        vectors = [
            {
                "sample_token": token,
                "offset_bytes": index * _VECTOR_BYTES,
                "byte_size": _VECTOR_BYTES,
                "sha256": _mapped_sha256(
                    mapping, offset=index * _VECTOR_BYTES, byte_size=_VECTOR_BYTES
                ),
            }
            for index, token in enumerate(tokens)
        ]
        _verify_regular_mapping_unchanged(descriptor_fd, path, initial)
    finally:
        del matrix
        mapping.close()
        os.close(descriptor_fd)
    descriptor = {
        "schema_version": CACHE_SCHEMA,
        "successor_id": successor_id,
        "successor_inventory_bundle_sha256": successor_inventory_bundle_sha256,
        "successor_inventory_sha256": successor_inventory_sha256,
        "evaluation_panel_sha256": evaluation_panel_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_manifest_sha256": preprocessing_manifest_sha256,
        "embedding_manifest_sha256": embedding_manifest_sha256,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "dtype": "float32_little_endian",
        "normalization": "L2",
        "pack_path": os.fspath(path),
        "pack_byte_size": expected_size,
        "pack_sha256": pack_sha256,
        "sample_tokens": tokens,
        "sample_tokens_sha256": content_sha256(tokens),
        "vectors": vectors,
    }
    return {**descriptor, "cache_descriptor_sha256": content_sha256(descriptor)}


def open_successor_embedding_cache(
    descriptor: object,
    *,
    successor_inventory_bundle: object,
    evaluation_panel: object,
) -> ValidatedSuccessorEmbeddingCache:
    """Rehash and validate an exact 128D float32 L2 cache before use."""

    inventory = validate_face_visible_successor_inventory_bundle(
        successor_inventory_bundle, verify_artifacts=False
    )
    panel = validate_fixed_evaluation_panel(evaluation_panel)
    value = _validate_cache_descriptor(descriptor)
    expected = {
        "successor_inventory_bundle_sha256": inventory["bundle_sha256"],
        "successor_inventory_sha256": inventory["inventory_sha256"],
        "evaluation_panel_sha256": panel["panel_sha256"],
    }
    for field, item in expected.items():
        if value[field] != item:
            raise Full128SuccessorEvaluationError(f"cache {field} differs")
    if value["sample_tokens"] != panel["required_sample_tokens"]:
        raise Full128SuccessorEvaluationError(
            "cache population must exactly equal the fixed evaluation panel"
        )
    auxiliary = {
        row["sample_token"]
        for row in inventory["inventory"]["identity_free_auxiliary_population"]
    }
    if auxiliary & set(value["sample_tokens"]):
        raise Full128SuccessorEvaluationError(
            "identity-free auxiliary samples leaked into successor cache"
        )
    path = _regular_file(Path(value["pack_path"]), "successor embedding pack")
    descriptor_fd, mapping, initial = _open_regular_mapping(
        path, expected_size=value["pack_byte_size"]
    )
    matrix = np.ndarray(
        (len(value["sample_tokens"]), EMBEDDING_DIMENSION),
        dtype="<f4",
        buffer=mapping,
    )
    matrix.setflags(write=False)
    try:
        if _mapped_sha256(mapping) != value["pack_sha256"]:
            raise Full128SuccessorEvaluationError(
                "successor embedding pack was tampered with"
            )
        _validate_matrix(matrix, len(value["sample_tokens"]))
        for index, row in enumerate(value["vectors"]):
            if (
                _mapped_sha256(
                    mapping, offset=index * _VECTOR_BYTES, byte_size=_VECTOR_BYTES
                )
                != row["sha256"]
            ):
                raise Full128SuccessorEvaluationError(
                    f"successor cache vector digest differs: {row['sample_token']}"
                )
        _verify_regular_mapping_unchanged(descriptor_fd, path, initial)
    except BaseException:
        del matrix
        mapping.close()
        raise
    finally:
        os.close(descriptor_fd)
    return ValidatedSuccessorEmbeddingCache(value, path, matrix, mapping)


def paired_identity_cluster_bootstrap(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    resamples: int = 1_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a paired whole-identity percentile interval for left minus right."""

    _nonempty(metric, "paired bootstrap metric")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise Full128SuccessorEvaluationError("bootstrap resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Full128SuccessorEvaluationError("bootstrap seed must be non-negative")
    left = _paired_rows(left_rows, metric=metric)
    right = _paired_rows(right_rows, metric=metric)
    if set(left) != set(right):
        raise Full128SuccessorEvaluationError(
            "paired bootstrap query populations differ"
        )
    clusters: defaultdict[str, list[float]] = defaultdict(list)
    for token in sorted(left):
        left_cluster, left_value = left[token]
        right_cluster, right_value = right[token]
        if left_cluster != right_cluster:
            raise Full128SuccessorEvaluationError(
                "paired bootstrap identity clusters differ"
            )
        clusters[left_cluster].append(left_value - right_value)
    if len(clusters) < 2:
        raise Full128SuccessorEvaluationError(
            "paired bootstrap requires at least two identity clusters"
        )
    sums = np.asarray([math.fsum(values) for values in clusters.values()])
    counts = np.asarray([len(values) for values in clusters.values()])
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        estimates[index] = sums[sampled].sum() / counts[sampled].sum()
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    payload = {
        "schema_version": PAIRED_BOOTSTRAP_SCHEMA,
        "metric": metric,
        "estimate": float(sums.sum() / counts.sum()),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "confidence_level": 0.95,
        "cluster_unit": "registered_identity_id",
        "cluster_count": len(clusters),
        "paired_query_count": len(left),
        "resamples": resamples,
        "seed": seed,
    }
    return {**payload, "bootstrap_sha256": content_sha256(payload)}


def build_dev_selection_receipt(
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    objective_metric: str = "Rank-1",
) -> dict[str, Any]:
    """Select exactly once from DEV aggregates; CAL/exposed inputs are rejected."""

    if objective_metric not in _METRICS:
        raise Full128SuccessorEvaluationError("unsupported DEV selection objective")
    rows: list[dict[str, Any]] = []
    for result in candidate_results:
        if result.get("scope") != "DEV":
            raise Full128SuccessorEvaluationError(
                "DEV selection receipt cannot consume CAL or exposed results"
            )
        successor_id = result.get("successor_id")
        _nonempty(successor_id, "DEV candidate successor_id")
        value = _finite(
            result.get("metrics", {}).get(objective_metric), objective_metric
        )
        rows.append(
            {
                "successor_id": successor_id,
                "result_sha256": result.get("result_sha256"),
                "objective_value": value,
                "denominator": result.get("query_count"),
            }
        )
    if not rows or len({row["successor_id"] for row in rows}) != len(rows):
        raise Full128SuccessorEvaluationError(
            "DEV selection candidates must be non-empty and unique"
        )
    for row in rows:
        _sha(row["result_sha256"], "DEV candidate result")
        if (
            isinstance(row["denominator"], bool)
            or not isinstance(row["denominator"], int)
            or row["denominator"] <= 0
        ):
            raise Full128SuccessorEvaluationError(
                "DEV candidate denominator must be positive"
            )
    rows.sort(key=lambda item: item["successor_id"])
    selected = min(
        rows, key=lambda item: (-item["objective_value"], item["successor_id"])
    )
    payload = {
        "schema_version": DEV_SELECTION_SCHEMA,
        "selection_scope": "DEV_ONLY",
        "objective_metric": objective_metric,
        "tie_policy": "SUCCESSOR_ID_ASC",
        "candidates": rows,
        "selected_successor_id": selected["successor_id"],
        "calibration_scope_used": False,
        "exposed_scope_used": False,
    }
    return {**payload, "receipt_sha256": content_sha256(payload)}


def evaluate_successor_family(
    *,
    successor_inventory_bundle: object,
    source_panel: object | None = None,
    caches: Sequence[ValidatedSuccessorEmbeddingCache],
    gallery_root: Path,
    face_protocol_v2_bundle: object | None = None,
    gallery_query_panel_bundle: object | None = None,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate fixed cohorts and return private evidence plus a public aggregate."""

    inventory = validate_face_visible_successor_inventory_bundle(
        successor_inventory_bundle, verify_artifacts=False
    )
    panel = build_score_blind_fixed_evaluation_panel(
        inventory,
        source_panel,
        face_protocol_v2_bundle=face_protocol_v2_bundle,
        gallery_query_panel_bundle=gallery_query_panel_bundle,
    )
    if not caches:
        raise Full128SuccessorEvaluationError("successor family cache list is empty")
    by_successor: dict[str, ValidatedSuccessorEmbeddingCache] = {}
    for cache in caches:
        if not isinstance(cache, ValidatedSuccessorEmbeddingCache):
            raise TypeError("successor cache must be validated before evaluation")
        successor_id = cache.descriptor["successor_id"]
        if successor_id in by_successor:
            raise Full128SuccessorEvaluationError("successor family repeats a cache")
        if cache.descriptor["evaluation_panel_sha256"] != panel["panel_sha256"]:
            raise Full128SuccessorEvaluationError(
                "successor cache was not built for the effective fixed panel"
            )
        by_successor[successor_id] = cache
    root = Path(os.path.abspath(os.fspath(gallery_root)))
    if root.exists() or root.is_symlink():
        raise FileExistsError("refusing to overwrite successor gallery root")
    root.parent.resolve(strict=True)
    root.mkdir(mode=0o700)

    candidate_reports: list[dict[str, Any]] = []
    for successor_id in sorted(by_successor):
        candidate_reports.append(
            _evaluate_candidate(
                successor_id,
                by_successor[successor_id],
                panel=panel,
                gallery_root=root / successor_id,
            )
        )
    dev_results = [
        scope
        for report in candidate_reports
        for scope in report["scope_aggregates"]
        if scope["scope"] == "DEV"
    ]
    selection = build_dev_selection_receipt(dev_results)
    selected_id = selection["selected_successor_id"]
    selected_report = next(
        report for report in candidate_reports if report["successor_id"] == selected_id
    )
    paired: list[dict[str, Any]] = []
    for report in candidate_reports:
        if report["successor_id"] == selected_id:
            continue
        for scope in ("DEV", "CAL", "EXPOSED_DIAGNOSTIC"):
            left = _scope_query_rows(selected_report, scope)
            right = _scope_query_rows(report, scope)
            if left and right:
                paired.append(
                    {
                        "scope": scope,
                        "left_successor_id": selected_id,
                        "right_successor_id": report["successor_id"],
                        "intervals": [
                            paired_identity_cluster_bootstrap(
                                left,
                                right,
                                metric=metric,
                                resamples=bootstrap_resamples,
                                seed=bootstrap_seed,
                            )
                            for metric in _METRICS
                        ],
                    }
                )
    private_payload = {
        "schema_version": PRIVATE_REPORT_SCHEMA,
        "visibility": "PRIVATE",
        "successor_inventory_bundle_sha256": inventory["bundle_sha256"],
        "evaluation_panel": panel,
        "candidates": candidate_reports,
        "dev_selection_receipt": selection,
        "paired_identity_cluster_bootstrap": paired,
        "scope_interpretation": {
            "DEV": "MODEL_SELECTION_ONLY",
            "CAL": "CALIBRATION_REPORTING;NOT_SELECTION",
            "EXPOSED_DIAGNOSTIC": "RETROSPECTIVE_EXPOSED;NOT_FINAL_EVALUATION",
        },
    }
    private = {
        **private_payload,
        "report_sha256": content_sha256(private_payload),
    }
    return private, sanitize_successor_evaluation_report(private)


def evaluate_authoritative_successor_family(
    *,
    successor_inventory_bundle: object,
    face_protocol_v2_bundle: object,
    gallery_query_panel_bundle: object,
    caches: Sequence[ValidatedSuccessorEmbeddingCache],
    gallery_root: Path,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate directly from governance v2 and its persisted authoritative panel."""

    source_panel = build_authoritative_face_visible_panel(
        face_protocol_v2_bundle, gallery_query_panel_bundle
    )
    return evaluate_successor_family(
        successor_inventory_bundle=successor_inventory_bundle,
        source_panel=source_panel,
        caches=caches,
        gallery_root=gallery_root,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def sanitize_successor_evaluation_report(value: object) -> dict[str, Any]:
    """Publish aggregate results only; reject arbitrary/private report additions."""

    expected = {
        "schema_version",
        "visibility",
        "successor_inventory_bundle_sha256",
        "evaluation_panel",
        "candidates",
        "dev_selection_receipt",
        "paired_identity_cluster_bootstrap",
        "scope_interpretation",
        "report_sha256",
    }
    _keys(value, expected, "private successor report")
    report = dict(value)
    payload = {key: item for key, item in report.items() if key != "report_sha256"}
    if (
        report["schema_version"]
        not in {LEGACY_PRIVATE_REPORT_SCHEMA, PRIVATE_REPORT_SCHEMA}
        or report["visibility"] != "PRIVATE"
        or report["report_sha256"] != content_sha256(payload)
    ):
        raise Full128SuccessorEvaluationError(
            "private successor report was tampered with"
        )
    candidates = []
    for candidate in report["candidates"]:
        _keys(
            candidate,
            {
                "successor_id",
                "cache_descriptor_sha256",
                "cohort_results",
                "scope_aggregates",
                "gallery_bindings",
                "ranked_private_qkv_traces",
                "candidate_report_sha256",
            },
            "private successor candidate report",
        )
        candidate_payload = {
            key: item
            for key, item in candidate.items()
            if key != "candidate_report_sha256"
        }
        if candidate["candidate_report_sha256"] != content_sha256(candidate_payload):
            raise Full128SuccessorEvaluationError(
                "private successor candidate report was tampered with"
            )
        candidates.append(
            {
                "successor_id": candidate["successor_id"],
                "cache_descriptor_sha256": candidate["cache_descriptor_sha256"],
                "scope_aggregates": [
                    _sanitize_scope_aggregate(item)
                    for item in candidate["scope_aggregates"]
                ],
                "gallery_bindings": [
                    {
                        key: artifact[key]
                        for key in (
                            "scope",
                            "dataset_name",
                            "enrollment_k",
                            "gallery_sha256",
                            "scorer_hash",
                            "template_count",
                            "identity_count",
                        )
                    }
                    for artifact in candidate["gallery_bindings"]
                ],
            }
        )
    public_payload = {
        "schema_version": PUBLIC_REPORT_SCHEMA,
        "visibility": "PUBLIC_AGGREGATE",
        "source_private_report_sha256": report["report_sha256"],
        "evaluation_panel_sha256": report["evaluation_panel"]["panel_sha256"],
        "candidates": candidates,
        "dev_selection_receipt": _sanitize_selection_receipt(
            report["dev_selection_receipt"]
        ),
        "paired_identity_cluster_bootstrap": [
            _sanitize_paired_result(item)
            for item in report["paired_identity_cluster_bootstrap"]
        ],
        "scope_interpretation": report["scope_interpretation"],
        "contains_embeddings": False,
        "contains_sample_or_identity_tokens": False,
        "contains_ranked_qkv_traces": False,
        "limitations": list(_PUBLIC_LIMITATIONS),
    }
    _reject_private_public_keys(public_payload)
    return {**public_payload, "public_report_sha256": content_sha256(public_payload)}


def build_multiseed_terminal_successor_decision(
    report_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the precommitted per-seed B5-SPATIAL versus B3 DEV gate."""

    if len(report_sources) != 3:
        raise Full128SuccessorEvaluationError(
            "terminal decision requires exactly three seed reports"
        )
    seeds: list[dict[str, Any]] = []
    panel_sha256: str | None = None
    for source in report_sources:
        _keys(
            source,
            {
                "seed_index",
                "report",
                "raw_sha256",
                "canonical_payload_sha256",
                "byte_size",
            },
            "terminal decision report source",
        )
        seed_index = source["seed_index"]
        if isinstance(seed_index, bool) or not isinstance(seed_index, int):
            raise Full128SuccessorEvaluationError("seed index must be an integer")
        _sha(source["raw_sha256"], "source report raw digest")
        _sha(source["canonical_payload_sha256"], "source report canonical digest")
        if (
            isinstance(source["byte_size"], bool)
            or not isinstance(source["byte_size"], int)
            or source["byte_size"] <= 0
        ):
            raise Full128SuccessorEvaluationError(
                "source report byte size must be positive"
            )
        report = validate_public_successor_evaluation_report(source["report"])
        if source["canonical_payload_sha256"] != content_sha256(report):
            raise Full128SuccessorEvaluationError(
                "source report canonical payload digest differs"
            )
        if panel_sha256 is None:
            panel_sha256 = report["evaluation_panel_sha256"]
        elif report["evaluation_panel_sha256"] != panel_sha256:
            raise Full128SuccessorEvaluationError(
                "terminal seed reports use different evaluation panels"
            )

        receipt = report["dev_selection_receipt"]
        selection = next(
            row
            for row in receipt["candidates"]
            if row["successor_id"] == receipt["selected_successor_id"]
        )
        if selection["successor_id"] != "B5-SPATIAL":
            raise Full128SuccessorEvaluationError(
                "terminal seed report did not select B5-SPATIAL on DEV"
            )
        b5_candidate = next(
            candidate
            for candidate in report["candidates"]
            if candidate["successor_id"] == "B5-SPATIAL"
        )
        b3 = next(
            (row for row in receipt["candidates"] if row["successor_id"] == "B3"),
            None,
        )
        if b3 is None:
            raise Full128SuccessorEvaluationError(
                "terminal seed report omits B3 from DEV selection"
            )
        comparison = next(
            (
                item
                for item in report["paired_identity_cluster_bootstrap"]
                if item["scope"] == "DEV"
                and item["left_successor_id"] == "B5-SPATIAL"
                and item["right_successor_id"] == "B3"
            ),
            None,
        )
        if comparison is None:
            raise Full128SuccessorEvaluationError(
                "terminal seed report omits precommitted DEV comparison"
            )
        interval = next(
            item for item in comparison["intervals"] if item["metric"] == "Rank-1"
        )
        expected_estimate = selection["objective_value"] - b3["objective_value"]
        if not math.isclose(
            interval["estimate"], expected_estimate, rel_tol=0.0, abs_tol=1e-15
        ):
            raise Full128SuccessorEvaluationError(
                "paired DEV estimate differs from selected point estimates"
            )
        seeds.append(
            {
                "seed_index": seed_index,
                "report_recorded_seed": interval["seed"],
                "source_report_raw_sha256": source["raw_sha256"],
                "source_report_canonical_payload_sha256": source[
                    "canonical_payload_sha256"
                ],
                "source_report_byte_size": source["byte_size"],
                "source_public_report_sha256": report["public_report_sha256"],
                "source_private_report_sha256": report["source_private_report_sha256"],
                "source_b5_spatial_cache_descriptor_sha256": b5_candidate[
                    "cache_descriptor_sha256"
                ],
                "dev_selection": {
                    "receipt_sha256": receipt["receipt_sha256"],
                    "scope": "DEV",
                    "role": "SINGLE_REPORT_POINT_SELECTION;NOT_SCIENTIFIC_PROMOTION",
                    "objective_metric": "Rank-1",
                    "selected_successor_id": selection["successor_id"],
                    "selected_point_estimate": selection["objective_value"],
                    "denominator": selection["denominator"],
                    "result_sha256": selection["result_sha256"],
                },
                "precommitted_gate": {
                    "comparison": "B5-SPATIAL_MINUS_B3",
                    "scope": "DEV",
                    "metric": "Rank-1",
                    "confidence_level": interval["confidence_level"],
                    "estimate": interval["estimate"],
                    "lower_bound": interval["lower_bound"],
                    "upper_bound": interval["upper_bound"],
                    "cluster_unit": interval["cluster_unit"],
                    "cluster_count": interval["cluster_count"],
                    "paired_query_count": interval["paired_query_count"],
                    "resamples": interval["resamples"],
                    "bootstrap_sha256": interval["bootstrap_sha256"],
                    "passes_strict_positive_lower_bound": interval["lower_bound"] > 0,
                },
            }
        )
    seeds.sort(key=lambda item: item["seed_index"])
    if [item["seed_index"] for item in seeds] != [0, 1, 2]:
        raise Full128SuccessorEvaluationError(
            "terminal decision seed indexes must be exactly 0, 1, and 2"
        )
    if len({item["source_report_raw_sha256"] for item in seeds}) != 3:
        raise Full128SuccessorEvaluationError(
            "terminal decision requires three distinct raw source reports"
        )
    if len({item["source_report_canonical_payload_sha256"] for item in seeds}) != 3:
        raise Full128SuccessorEvaluationError(
            "terminal decision requires three distinct canonical source reports"
        )
    if len({item["source_public_report_sha256"] for item in seeds}) != 3:
        raise Full128SuccessorEvaluationError(
            "terminal decision requires three distinct public reports"
        )
    if len({item["report_recorded_seed"] for item in seeds}) != 3:
        raise Full128SuccessorEvaluationError(
            "terminal decision requires three distinct recorded bootstrap seeds"
        )
    if len({item["source_b5_spatial_cache_descriptor_sha256"] for item in seeds}) != 3:
        raise Full128SuccessorEvaluationError(
            "terminal decision requires three distinct B5-SPATIAL cache descriptors"
        )
    selected_values = [
        item["dev_selection"]["selected_point_estimate"] for item in seeds
    ]
    paired_estimates = [item["precommitted_gate"]["estimate"] for item in seeds]
    failed = [
        item["seed_index"]
        for item in seeds
        if not item["precommitted_gate"]["passes_strict_positive_lower_bound"]
    ]
    decision = "NO_GO" if failed else "GO"
    payload = {
        "schema_version": TERMINAL_DECISION_SCHEMA,
        "visibility": "PUBLIC_DECISION",
        "evaluation_panel_sha256": panel_sha256,
        "seed_reports": seeds,
        "across_seed_summaries": {
            "interpretation": (
                "DESCRIPTIVE_ACROSS_TRAINING_SEEDS_ONLY;NO_ACROSS_SEED_CI;"
                "NO_META_ANALYSIS;NO_POOLED_QUERY_CLAIM"
            ),
            "selected_dev_rank1": _descriptive_summary(selected_values),
            "paired_dev_rank1_difference": _descriptive_summary(paired_estimates),
        },
        "promotion_gate": {
            "precommitted_rule": (
                "GO_IFF_EACH_SEED_B5_SPATIAL_MINUS_B3_DEV_RANK1_"
                "PAIRED_95CI_LOWER_BOUND_GT_0"
            ),
            "required_seed_count": 3,
            "passing_seed_count": 3 - len(failed),
            "failed_seed_indexes": failed,
            "minimum_lower_bound": min(
                item["precommitted_gate"]["lower_bound"] for item in seeds
            ),
            "decision": decision,
            "scientific_promotion": (
                "REJECT_B5_SPATIAL_OVER_B3"
                if decision == "NO_GO"
                else "PROMOTE_B5_SPATIAL_OVER_B3_WITHIN_DEV_MODEL_SELECTION_ONLY"
            ),
        },
        "evidence_boundaries": {
            "single_report_dev_selection": (
                "POINT_SELECTION_RECEIPT_ONLY;NOT_SCIENTIFIC_PROMOTION"
            ),
            "calibration": {
                "selection_input_used": False,
                "role": "CALIBRATION_REPORTING_ONLY;NOT_SELECTION_OR_PROMOTION",
            },
            "exposed_diagnostic": {
                "selection_input_used": False,
                "role": "RETROSPECTIVE_EXPOSED_ONLY;NOT_SELECTION_OR_FINAL_EVALUATION",
            },
            "independent_final": {
                "availability": "UNAVAILABLE",
                "decision_input_used": False,
                "claim": "NO_INDEPENDENT_FINAL_PERFORMANCE_OR_GENERALIZATION_CLAIM",
            },
            "input_privacy": {
                "accepted_input": "STRICTLY_VALIDATED_PUBLIC_AGGREGATE_REPORTS_ONLY",
                "contains_private_paths_or_tokens": False,
                "contains_embeddings_or_ranked_traces": False,
            },
            "run_seed_binding": {
                "available_evidence": "REPORT_RECORDED_BOOTSTRAP_SEED",
                "limitation": (
                    "NO_SEPARATE_TRAINING_RUN_SEED_FIELD_IS_AVAILABLE_IN_THE_"
                    "PUBLIC_REPORT"
                ),
            },
        },
    }
    _reject_private_public_keys(payload)
    return {**payload, "decision_sha256": content_sha256(payload)}


def _evaluate_candidate(
    successor_id: str,
    cache: ValidatedSuccessorEmbeddingCache,
    *,
    panel: Mapping[str, Any],
    gallery_root: Path,
) -> dict[str, Any]:
    gallery_root.mkdir(mode=0o700)
    record_by_token = {row["sample_token"]: row for row in panel["records"]}
    vectors = cache.load_embeddings(panel["required_sample_tokens"])
    vector_by_token = dict(zip(panel["required_sample_tokens"], vectors, strict=True))
    results: list[dict[str, Any]] = []
    gallery_bindings: list[dict[str, Any]] = []
    private_traces: list[dict[str, Any]] = []
    for cohort in panel["cohorts"]:
        if cohort["status"] != "AVAILABLE":
            results.append(
                {
                    **cohort,
                    "metrics": None,
                    "query_rows": [],
                    "result_sha256": content_sha256(cohort),
                }
            )
            continue
        result, binding, traces = _evaluate_cohort(
            successor_id=successor_id,
            cohort=cohort,
            record_by_token=record_by_token,
            vector_by_token=vector_by_token,
            descriptor=cache.descriptor,
            gallery_directory=gallery_root
            / (f"{cohort['scope']}-{cohort['dataset_name']}-K{cohort['enrollment_k']}"),
        )
        results.append(result)
        gallery_bindings.append(binding)
        private_traces.extend(traces)
    scope_aggregates = [
        _scope_aggregate(successor_id, scope, results) for scope in EVALUATION_SCOPES
    ]
    payload = {
        "successor_id": successor_id,
        "cache_descriptor_sha256": cache.descriptor["cache_descriptor_sha256"],
        "cohort_results": results,
        "scope_aggregates": scope_aggregates,
        "gallery_bindings": gallery_bindings,
        "ranked_private_qkv_traces": private_traces,
    }
    return {**payload, "candidate_report_sha256": content_sha256(payload)}


def _evaluate_cohort(
    *,
    successor_id: str,
    cohort: Mapping[str, Any],
    record_by_token: Mapping[str, Mapping[str, Any]],
    vector_by_token: Mapping[str, np.ndarray],
    descriptor: Mapping[str, Any],
    gallery_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from retrieval.gallery import (
        GalleryEnrollment,
        IdentityGallery,
        IdentityRegistryPolicy,
    )
    from retrieval.qkv import EnrollmentRank

    query_tokens = cohort["query_sample_tokens"]
    gallery_tokens = cohort["gallery_sample_tokens"]
    identities = frozenset(
        record_by_token[token]["registered_identity_id"] for token in gallery_tokens
    )
    contract = _gallery_contract(descriptor)
    rank = EnrollmentRank(f"K{cohort['enrollment_k']}")
    enrollments = [
        GalleryEnrollment(
            embedding=vector_by_token[token],
            registered_identity_id=record_by_token[token]["registered_identity_id"],
            metadata={"sample_token": token, "scope": cohort["scope"]},
            idempotency_key=f"{successor_id}:{cohort['scope']}:{cohort['enrollment_k']}:{token}",
            content_sha256=hashlib.sha256(
                descriptor["cache_descriptor_sha256"].encode("ascii")
                + token.encode("ascii")
                + vector_by_token[token].astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            enrollment_rank=rank,
            enrollment_view=token,
            duplicate_group_ids=(record_by_token[token]["duplicate_component"],),
        )
        for token in gallery_tokens
    ]
    policy = IdentityRegistryPolicy(registered_identity_ids=identities)
    built = IdentityGallery.build(
        gallery_directory,
        enrollments,
        dim=EMBEDDING_DIMENSION,
        embedding_contract=contract,
        registry_policy=policy,
    )
    built.close()
    gallery = IdentityGallery(
        gallery_directory,
        dim=EMBEDDING_DIMENSION,
        embedding_contract=contract,
        read_only=True,
        registry_policy=policy,
    )
    query_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    try:
        for token in query_tokens:
            expected_identity = record_by_token[token]["registered_identity_id"]
            ranked = gallery.search(vector_by_token[token], top_k=len(identities))
            ranked_ids = [item[2]["registered_dog_id"] for item in ranked]
            if expected_identity not in ranked_ids:
                raise Full128SuccessorEvaluationError(
                    "fixed closed-set query lacks a relevant reopened gallery identity"
                )
            relevant_rank = ranked_ids.index(expected_identity) + 1
            row = {
                "sample_token": token,
                "bootstrap_cluster_id": expected_identity,
                "relevant_rank": relevant_rank,
                "Rank-1": float(relevant_rank <= 1),
                "Rank-5": float(relevant_rank <= 5),
                "Rank-10": float(relevant_rank <= 10),
                "MRR": 1.0 / relevant_rank,
            }
            query_rows.append(row)
            traces.append(
                {
                    "scope": cohort["scope"],
                    "dataset_name": cohort["dataset_name"],
                    "enrollment_k": cohort["enrollment_k"],
                    "Q": {
                        "sample_token": token,
                    },
                    "ranked_KV": [
                        {
                            "rank": index + 1,
                            "score": float(item[1]),
                            "K": {
                                "winning_template_row": item[0],
                                "sample_token": item[2]["metadata"]["sample_token"],
                            },
                            "V": {
                                "registered_identity_id": item[2]["registered_dog_id"],
                                "template_id": item[2]["template_id"],
                                "content_sha256": item[2]["content_sha256"],
                            },
                        }
                        for index, item in enumerate(ranked)
                    ],
                    "exact_cosine": True,
                }
            )
        scorer_hash = gallery.scorer_hash
    finally:
        gallery.close()
    metrics = {
        metric: float(np.mean([row[metric] for row in query_rows]))
        for metric in _METRICS
    }
    base = {
        "scope": cohort["scope"],
        "dataset_name": cohort["dataset_name"],
        "enrollment_k": cohort["enrollment_k"],
        "status": "AVAILABLE",
        "reason": None,
        "query_count": len(query_rows),
        "gallery_template_count": len(gallery_tokens),
        "gallery_identity_count": len(identities),
        "metrics": metrics,
        "query_rows": query_rows,
    }
    result = {**base, "result_sha256": content_sha256(base)}
    gallery_sha = _directory_sha256(gallery_directory)
    binding = {
        "scope": cohort["scope"],
        "dataset_name": cohort["dataset_name"],
        "enrollment_k": cohort["enrollment_k"],
        "gallery_sha256": gallery_sha,
        "scorer_hash": scorer_hash,
        "template_count": len(gallery_tokens),
        "identity_count": len(identities),
        "reopened_read_only": True,
        "exact_cosine": True,
    }
    return result, binding, traces


def _scope_aggregate(
    successor_id: str, scope: str, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [
        row
        for result in results
        if result["scope"] == scope
        and result["enrollment_k"] == 1
        and result["status"] == "AVAILABLE"
        for row in result["query_rows"]
    ]
    if not rows:
        base = {
            "successor_id": successor_id,
            "scope": scope,
            "status": "NOT_AVAILABLE",
            "reason": "no K-feasible fixed-panel query remains",
            "query_count": 0,
            "identity_count": 0,
            "metrics": {metric: None for metric in _METRICS},
        }
    else:
        base = {
            "successor_id": successor_id,
            "scope": scope,
            "status": "AVAILABLE",
            "reason": None,
            "query_count": len(rows),
            "identity_count": len({row["bootstrap_cluster_id"] for row in rows}),
            "metrics": {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in _METRICS
            },
        }
    return {**base, "result_sha256": content_sha256(base)}


def _scope_query_rows(report: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    # One fixed K avoids counting the same query three times in paired intervals.
    return [
        row
        for result in report["cohort_results"]
        if result["scope"] == scope
        and result["enrollment_k"] == 1
        and result["status"] == "AVAILABLE"
        for row in result["query_rows"]
    ]


def _panel_cohorts(
    records: Sequence[Mapping[str, Any]], *, membership_contract: str
) -> list[dict[str, Any]]:
    cohorts: list[dict[str, Any]] = []
    for scope in EVALUATION_SCOPES:
        scoped = [row for row in records if row["scope"] == scope]
        for dataset_name in sorted({row["dataset_name"] for row in scoped}):
            dataset_rows = [
                row for row in scoped if row["dataset_name"] == dataset_name
            ]
            for k in ENROLLMENT_KS:
                query = sorted(
                    row["sample_token"]
                    for row in dataset_rows
                    if row["uses_by_enrollment_k"][str(k)]["use"] == "QUERY"
                )
                gallery = sorted(
                    row["sample_token"]
                    for row in dataset_rows
                    if row["uses_by_enrollment_k"][str(k)]["use"] == "GALLERY"
                )
                gallery_identities = {
                    row["registered_identity_id"]
                    for row in dataset_rows
                    if row["sample_token"] in gallery
                }
                query_identities = {
                    row["registered_identity_id"]
                    for row in dataset_rows
                    if row["sample_token"] in query
                }
                feasible = bool(
                    query and gallery and query_identities <= gallery_identities
                )
                if (
                    feasible
                    and membership_contract == "AUTHORITATIVE_SHARED_QUERY_NESTED_K"
                ):
                    query_counts: defaultdict[str, int] = defaultdict(int)
                    gallery_counts: defaultdict[str, int] = defaultdict(int)
                    for row in dataset_rows:
                        use = row["uses_by_enrollment_k"][str(k)]["use"]
                        if use == "QUERY":
                            query_counts[row["registered_identity_id"]] += 1
                        elif use == "GALLERY":
                            gallery_counts[row["registered_identity_id"]] += 1
                    expected_identities = {
                        row["registered_identity_id"]
                        for row in dataset_rows
                        if row["authoritative_cohort_member"]
                    }
                    feasible = bool(expected_identities) and all(
                        query_counts[identity] == 1 and gallery_counts[identity] == k
                        for identity in expected_identities
                    )
                cohorts.append(
                    {
                        "scope": scope,
                        "dataset_name": dataset_name,
                        "enrollment_k": k,
                        "status": "AVAILABLE" if feasible else "NOT_AVAILABLE",
                        "reason": None
                        if feasible
                        else "fixed panel has no closed-set K-feasible population",
                        "query_sample_tokens": query if feasible else [],
                        "gallery_sample_tokens": gallery if feasible else [],
                        "terminal_exclusion_count": sum(
                            row["uses_by_enrollment_k"][str(k)]["use"]
                            == "TERMINAL_EXCLUSION"
                            for row in dataset_rows
                        ),
                    }
                )
    return cohorts


def _validate_evaluation_panel_record(value: object, *, authoritative: bool) -> None:
    expected = {
        "sample_token",
        "registered_identity_id",
        "dataset_name",
        "scope",
        "duplicate_component",
        "source_panel_record_sha256",
        "successor_inventory_record_sha256",
        "uses_by_enrollment_k",
    }
    if authoritative:
        expected.add("authoritative_cohort_member")
    _keys(
        value,
        expected,
        "successor evaluation panel record",
    )
    for field in (
        "sample_token",
        "duplicate_component",
        "source_panel_record_sha256",
        "successor_inventory_record_sha256",
    ):
        _sha(value[field], field)
    _nonempty(value["registered_identity_id"], "registered_identity_id")
    _nonempty(value["dataset_name"], "dataset_name")
    if authoritative and not isinstance(value["authoritative_cohort_member"], bool):
        raise Full128SuccessorEvaluationError(
            "evaluation panel authoritative membership must be boolean"
        )
    if value["scope"] not in (*EVALUATION_SCOPES, "FIT"):
        raise Full128SuccessorEvaluationError("evaluation panel scope differs")
    uses = value["uses_by_enrollment_k"]
    if not isinstance(uses, Mapping) or set(uses) != {str(k) for k in ENROLLMENT_KS}:
        raise Full128SuccessorEvaluationError("evaluation panel K uses differ")
    for use in uses.values():
        _keys(use, {"use", "reason"}, "evaluation panel use")
        if use["use"] not in {
            "GALLERY",
            "QUERY",
            "TERMINAL_EXCLUSION",
            "TRAINING_ONLY",
        }:
            raise Full128SuccessorEvaluationError("evaluation panel use differs")
        if (use["use"] in {"GALLERY", "QUERY"}) != (use["reason"] is None):
            raise Full128SuccessorEvaluationError("evaluation panel use reason differs")


def _reject_evaluation_panel_leakage(records: Sequence[Mapping[str, Any]]) -> None:
    for scope in EVALUATION_SCOPES:
        scoped = [row for row in records if row["scope"] == scope]
        for k in ENROLLMENT_KS:
            query = [
                row
                for row in scoped
                if row["uses_by_enrollment_k"][str(k)]["use"] == "QUERY"
            ]
            gallery = [
                row
                for row in scoped
                if row["uses_by_enrollment_k"][str(k)]["use"] == "GALLERY"
            ]
            if {row["sample_token"] for row in query} & {
                row["sample_token"] for row in gallery
            } or {row["duplicate_component"] for row in query} & {
                row["duplicate_component"] for row in gallery
            }:
                raise Full128SuccessorEvaluationError(
                    f"fixed evaluation panel leakage: {scope}/K{k}"
                )


def _validate_cache_descriptor(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "successor_id",
        "successor_inventory_bundle_sha256",
        "successor_inventory_sha256",
        "evaluation_panel_sha256",
        "model_manifest_sha256",
        "checkpoint_sha256",
        "preprocessing_manifest_sha256",
        "embedding_manifest_sha256",
        "embedding_dimension",
        "dtype",
        "normalization",
        "pack_path",
        "pack_byte_size",
        "pack_sha256",
        "sample_tokens",
        "sample_tokens_sha256",
        "vectors",
        "cache_descriptor_sha256",
    }
    _keys(value, expected, "successor cache descriptor")
    descriptor = dict(value)
    payload = {
        key: item
        for key, item in descriptor.items()
        if key != "cache_descriptor_sha256"
    }
    if (
        descriptor["schema_version"] != CACHE_SCHEMA
        or descriptor["cache_descriptor_sha256"] != content_sha256(payload)
        or descriptor["embedding_dimension"] != EMBEDDING_DIMENSION
        or descriptor["dtype"] != "float32_little_endian"
        or descriptor["normalization"] != "L2"
    ):
        raise Full128SuccessorEvaluationError("successor cache contract differs")
    for field in (
        "successor_inventory_bundle_sha256",
        "successor_inventory_sha256",
        "evaluation_panel_sha256",
        "model_manifest_sha256",
        "checkpoint_sha256",
        "preprocessing_manifest_sha256",
        "embedding_manifest_sha256",
        "pack_sha256",
        "sample_tokens_sha256",
        "cache_descriptor_sha256",
    ):
        _sha(descriptor[field], field)
    tokens = descriptor["sample_tokens"]
    if (
        not isinstance(tokens, list)
        or not tokens
        or tokens != sorted(set(tokens))
        or descriptor["sample_tokens_sha256"] != content_sha256(tokens)
    ):
        raise Full128SuccessorEvaluationError("successor cache token order differs")
    if (
        descriptor["pack_byte_size"] != len(tokens) * _VECTOR_BYTES
        or descriptor["pack_byte_size"] > _MAX_PACK_BYTES
        or not isinstance(descriptor["pack_path"], str)
        or not Path(descriptor["pack_path"]).is_absolute()
    ):
        raise Full128SuccessorEvaluationError("successor cache storage differs")
    vectors = descriptor["vectors"]
    if not isinstance(vectors, list) or len(vectors) != len(tokens):
        raise Full128SuccessorEvaluationError("successor cache vector index differs")
    for index, (token, row) in enumerate(zip(tokens, vectors, strict=True)):
        _keys(
            row,
            {"sample_token", "offset_bytes", "byte_size", "sha256"},
            "successor cache vector",
        )
        if (
            row["sample_token"] != token
            or row["offset_bytes"] != index * _VECTOR_BYTES
            or row["byte_size"] != _VECTOR_BYTES
        ):
            raise Full128SuccessorEvaluationError(
                "successor cache vectors must be exact contiguous rows"
            )
        _sha(row["sha256"], "successor cache vector")
    return descriptor


def _validate_matrix(value: np.ndarray, rows: int) -> None:
    if value.dtype != np.dtype("float32") or value.shape != (rows, EMBEDDING_DIMENSION):
        raise Full128SuccessorEvaluationError("cache matrix must be exact 128D float32")
    rows_per_chunk = max(1, (1 << 20) // _VECTOR_BYTES)
    for start in range(0, rows, rows_per_chunk):
        chunk = value[start : start + rows_per_chunk]
        if not np.isfinite(chunk).all():
            raise Full128SuccessorEvaluationError(
                "cache matrix contains non-finite values"
            )
        norms = np.linalg.norm(chunk, axis=1)
        if not np.all(np.isclose(norms, 1.0, rtol=0.0, atol=1e-5)):
            raise Full128SuccessorEvaluationError(
                "cache matrix rows must be L2-normalized"
            )


def _open_regular_mapping(
    path: Path, *, expected_size: int
) -> tuple[int, mmap.mmap, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_size != expected_size
            or not 0 < expected_size <= _MAX_PACK_BYTES
        ):
            raise Full128SuccessorEvaluationError(
                "successor embedding pack byte size differs"
            )
        mapping = mmap.mmap(descriptor, expected_size, access=mmap.ACCESS_READ)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, mapping, initial


def _verify_regular_mapping_unchanged(
    descriptor: int, path: Path, initial: os.stat_result
) -> None:
    final = os.fstat(descriptor)
    named = os.stat(path, follow_symlinks=False)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(initial) != identity(final)
        or not stat.S_ISREG(named.st_mode)
        or (named.st_dev, named.st_ino) != (initial.st_dev, initial.st_ino)
    ):
        raise Full128SuccessorEvaluationError(
            "successor embedding pack changed while being validated"
        )


def _mapped_sha256(
    mapping: mmap.mmap, *, offset: int = 0, byte_size: int | None = None
) -> str:
    size = len(mapping) - offset if byte_size is None else byte_size
    if offset < 0 or size < 0 or offset + size > len(mapping):
        raise Full128SuccessorEvaluationError("mapped digest range differs")
    digest = hashlib.sha256()
    view = memoryview(mapping)
    try:
        stop = offset + size
        for start in range(offset, stop, 1 << 20):
            digest.update(view[start : min(start + (1 << 20), stop)])
    finally:
        view.release()
    return digest.hexdigest()


def _gallery_contract(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "cvi.gallery_embedding_contract.v1",
        "kind": "FULL128_FACE_VISIBLE_SUCCESSOR",
        "dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "normalization": "L2",
        "channels": [
            {"name": "Full128", "dimension": EMBEDDING_DIMENSION, "optional": False}
        ],
        "fusion": {
            "type": "exact_available_intersection_weighted_cosine.v1",
            "weights": [1.0],
            "exact": True,
        },
        "successor_binding": {
            field: descriptor[field]
            for field in (
                "successor_id",
                "model_manifest_sha256",
                "checkpoint_sha256",
                "preprocessing_manifest_sha256",
                "embedding_manifest_sha256",
                "successor_inventory_bundle_sha256",
                "successor_inventory_sha256",
                "evaluation_panel_sha256",
                "cache_descriptor_sha256",
            )
        },
    }


def _paired_rows(
    rows: Sequence[Mapping[str, Any]], *, metric: str
) -> dict[str, tuple[str, float]]:
    result: dict[str, tuple[str, float]] = {}
    for row in rows:
        token = row.get("sample_token")
        cluster = row.get("bootstrap_cluster_id")
        _nonempty(token, "paired query token")
        _nonempty(cluster, "paired identity cluster")
        if token in result:
            raise Full128SuccessorEvaluationError("paired rows repeat a query")
        result[token] = (cluster, _finite(row.get(metric), metric))
    if not result:
        raise Full128SuccessorEvaluationError("paired bootstrap rows are empty")
    return result


def _directory_sha256(path: Path) -> str:
    rows = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_file():
            raise Full128SuccessorEvaluationError("gallery contains an unsafe entry")
        payload = _read_regular(child, maximum=_MAX_PACK_BYTES, allow_empty=True)
        rows.append(
            {
                "name": child.name,
                "byte_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return content_sha256(rows)


def _regular_file(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.is_symlink() or not absolute.is_file():
        raise Full128SuccessorEvaluationError(f"{label} must be a regular file")
    return absolute


def _read_regular(path: Path, *, maximum: int, allow_empty: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    observed = 0
    try:
        before = os.fstat(descriptor)
        minimum = 0 if allow_empty else 1
        if not stat.S_ISREG(before.st_mode) or not minimum <= before.st_size <= maximum:
            raise Full128SuccessorEvaluationError(
                "bounded artifact size or type differs"
            )
        while chunk := os.read(descriptor, min(1_048_576, maximum + 1 - observed)):
            observed += len(chunk)
            if observed > maximum:
                raise Full128SuccessorEvaluationError("artifact exceeds byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or observed != before.st_size:
        raise Full128SuccessorEvaluationError("artifact changed while being read")
    return b"".join(chunks)


def _descriptive_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise Full128SuccessorEvaluationError(
            "across-seed summary requires at least two values"
        )
    parsed = [_finite(value, "across-seed value") for value in values]
    mean = math.fsum(parsed) / len(parsed)
    sample_variance = math.fsum((value - mean) ** 2 for value in parsed) / (
        len(parsed) - 1
    )
    return {
        "seed_count": len(parsed),
        "arithmetic_mean": mean,
        "sample_standard_deviation": math.sqrt(sample_variance),
        "minimum": min(parsed),
        "maximum": max(parsed),
    }


def _keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Full128SuccessorEvaluationError(f"{label} fields differ")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Full128SuccessorEvaluationError(f"{label} must be lowercase SHA-256")


def _nonempty(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise Full128SuccessorEvaluationError(f"{label} must be non-empty text")


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise Full128SuccessorEvaluationError(f"{label} must be finite")
    return float(value)


__all__ = [
    "AUTHORITATIVE_EVALUATION_PANEL_SCHEMA",
    "CACHE_SCHEMA",
    "DEV_SELECTION_SCHEMA",
    "EVALUATION_PANEL_SCHEMA",
    "LEGACY_PRIVATE_REPORT_SCHEMA",
    "PRIVATE_REPORT_SCHEMA",
    "PUBLIC_REPORT_SCHEMA",
    "TERMINAL_DECISION_SCHEMA",
    "Full128SuccessorEvaluationError",
    "ValidatedSuccessorEmbeddingCache",
    "build_authoritative_fixed_evaluation_panel",
    "build_dev_selection_receipt",
    "build_multiseed_terminal_successor_decision",
    "build_score_blind_fixed_evaluation_panel",
    "build_successor_embedding_cache_descriptor",
    "evaluate_authoritative_successor_family",
    "evaluate_successor_family",
    "open_successor_embedding_cache",
    "paired_identity_cluster_bootstrap",
    "sanitize_successor_evaluation_report",
    "validate_fixed_evaluation_panel",
    "validate_public_successor_evaluation_report",
]
