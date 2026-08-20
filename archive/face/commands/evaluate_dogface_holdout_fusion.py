"""Prepare and evaluate the sealed DogFace Appearance/Face holdout fusion."""

from __future__ import annotations

from archive.root import repository_root as find_repo_root
import argparse
import io
import json
import os
import re
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from data.public_sources.public_canine_manifest import DOGFACE_DATASET
from evaluation.calibration import (
    compute_probability_calibration_metrics,
    fit_isotonic_calibration,
)
from evaluation.embedding_diagnostics import compute_embedding_diagnostics
from evaluation.search_metrics.metrics import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
    identity_clustered_bootstrap_ci,
)
from evaluation.robustness_protocol import (
    RobustnessProtocolConfig,
    build_dataset_balanced_oof_protocol,
)
from representation.evidence.oof_simplex import OOFSimplexConfig, fit_oof_simplex
from shared.foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
    write_private_json_directory_bundle,
)
from shared.foundation.provenance import content_sha256
from enrollment.registry.identity_registry import (
    compute_identity_token,
    compute_public_subject_token,
    compute_registered_dog_id,
)
from evaluation.splits.role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    RoleExposureLedger,
    RoleExposureReceipt,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
    verify_role_exposure_receipt,
)
from evaluation.splits.split_role_exposure import verify_split_role_exposure_inputs
from identification.export.face.checkpoint import (
    expected_faceid_contract_for_checkpoint,
    normalize_dino_local_artifact_contract,
    validate_checkpoint_runtime_bindings,
    validate_checkpoint_structure,
)
from identification.training.face.trainer import (
    build_faceid_model,
    load_receipt_bound_frozen_dino,
)

if __package__:
    from archive.shared_helpers.commands import evaluate_external_appearance as external
else:  # pragma: no cover - exercised by source-checkout CLI invocation
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "shared_helpers" / "commands")
    )
    import evaluate_external_appearance as external


PLAN_SCHEMA = "cvi.dogface_holdout_fusion_plan.v1"
REPORT_SCHEMA = "cvi.dogface_holdout_fusion_evaluation.v1"
_CALIBRATION_ROLE = "DOGFACE_DEVELOPMENT"
_FINAL_ROLE = "DOGFACE_CALIBRATION"
_EXPECTED_IDENTITIES = 125
_FOLDS = 5
_OOF_SEED = 0
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_AGGREGATION_CANDIDATES = (
    {"id": "log_mean_exp_t0.05", "aggregation": "log_mean_exp", "temperature": 0.05},
    {"id": "log_mean_exp_t0.1", "aggregation": "log_mean_exp", "temperature": 0.1},
    {"id": "log_mean_exp_t0.2", "aggregation": "log_mean_exp", "temperature": 0.2},
    {"id": "log_mean_exp_t0.5", "aggregation": "log_mean_exp", "temperature": 0.5},
    {"id": "max", "aggregation": "max"},
    {"id": "mean", "aggregation": "mean"},
    {"id": "median", "aggregation": "median"},
    {"id": "top_k_mean_k2", "aggregation": "top_k_mean", "top_k": 2},
)
_CODE_PATHS = (
    "archive/face/commands/evaluate_dogface_holdout_fusion.py",
    "archive/shared_helpers/commands/evaluate_external_appearance.py",
    "archive/shared_helpers/commands/evaluate_roi_reid.py",
    "identification/training/appearance/trainer.py",
    "identification/training/appearance/config.py",
    "identification/export/appearance/evidencer.py",
    "shared/contracts/dinov2_contract.py",
    "identification/export/face/checkpoint.py",
    "identification/training/face/dataset.py",
    "identification/export/face/model.py",
    "identification/training/face/trainer.py",
    "archive/face/experiments/face_evaluation.py",
    "identification/training/nose/trainer.py",
    "evaluation/embedding_diagnostics.py",
    "evaluation/robustness_protocol.py",
    "evaluation/search_metrics/metrics.py",
    "evaluation/calibration.py",
    "representation/evidence/oof_simplex.py",
    "evaluation/splits/role_exposure.py",
    "shared/foundation/protected_io.py",
    "evaluation/splits/protected_public_split.py",
    "data/public_sources/public_canine_manifest.py",
    "data/public_sources/public_canine_semantic_intake.py",
    "data/public_sources/public_dataset_receipt_io.py",
    "evaluation/splits/training_admission.py",
    "enrollment/registry/identity_registry.py",
    "evaluation/splits/split_role_exposure.py",
    "evaluation/splits/split_registry_binding.py",
)
_FORBIDDEN_REPORT_KEYS = {
    "sample_token",
    "sample_tokens",
    "identity_token",
    "identity_tokens",
    "public_subject_token",
    "dataset_identity_id",
    "source_sample_id",
    "query_rows",
    "gallery_identity_order",
    "embeddings",
    "vectors",
    "scores",
}


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _parse_sha256(value: str) -> str:
    return _require_sha256(value, "command-line SHA-256")


def _repository_root() -> Path:
    return find_repo_root(__file__)


def _code_hashes(repository: Path | None = None) -> dict[str, str]:
    root = _repository_root() if repository is None else repository
    return {name: external._sha256_file((root / name).resolve()) for name in _CODE_PATHS}


def _verify_code_bindings(plan: Mapping[str, Any], repository: Path | None = None) -> None:
    expected = plan.get("code_sha256s")
    if not isinstance(expected, dict) or expected != _code_hashes(repository):
        raise ValueError("source code differs from the prepared plan")


def _refuse_existing(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite protected output: "
            + ", ".join(os.fspath(path) for path in existing)
        )


def _document_binding(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = read_strict_json_document(path)
    return document.payload, {
        "path": os.fspath(path),
        "raw_sha256": document.raw_sha256,
        "content_sha256": document.canonical_payload_sha256,
        "byte_size": document.byte_size,
    }


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": os.fspath(path),
        "sha256": external._sha256_file(path),
        "byte_size": path.stat(follow_symlinks=False).st_size,
    }


def _require_absolute_inputs(args: argparse.Namespace, names: Sequence[str]) -> None:
    for name in names:
        path = getattr(args, name)
        if not path.is_absolute():
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path")


def _load_authenticated_split(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    dict[str, Any],
    dict[str, Any],
]:
    assignment, assignment_binding = _document_binding(args.assignment)
    labels, labels_binding = _document_binding(args.labels)
    source_payload, source_binding = _document_binding(args.source_bundle)
    split_receipt, receipt_binding = _document_binding(args.split_receipt)
    source, source_by_token = external._validate_split_documents(
        assignment,
        labels,
        split_receipt,
        source_payload,
        expected_receipt_sha256=args.split_receipt_sha256,
    )
    if assignment.get("status") != "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        raise ValueError("protected split assignment did not pass")
    if assignment.get("score_inputs_used") is not False:
        raise ValueError("protected split assignment was not score-blind")
    return assignment, labels, source, source_by_token, {
        "assignment": assignment_binding,
        "labels": labels_binding,
        "source_bundle": source_binding,
        "split_receipt": receipt_binding,
    }


def _load_exposure_history(
    ledger_path: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    source: Any,
) -> tuple[RoleExposureLedger, RoleExposureReceipt, dict[str, Any]]:
    ledger_payload, ledger_binding = _document_binding(ledger_path)
    receipt_payload, receipt_binding = _document_binding(receipt_path)
    ledger = RoleExposureLedger.from_dict(ledger_payload)
    receipt = RoleExposureReceipt.from_dict(receipt_payload)
    verify_role_exposure_receipt(ledger, receipt)
    if receipt.receipt_sha256 != expected_receipt_sha256:
        raise ValueError("role exposure receipt differs from the external pin")
    verify_split_role_exposure_inputs(source.samples, ledger, receipt)
    return ledger, receipt, {
        "historical_exposure_ledger": ledger_binding,
        "historical_exposure_receipt": receipt_binding,
    }


def _build_holdout_populations(
    assignment: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    expected_identities: int = _EXPECTED_IDENTITIES,
) -> dict[str, Any]:
    label_by_sample = {
        record["sample_token"]: record for record in labels.get("records", [])
    }
    role_records: dict[str, list[Mapping[str, Any]]] = {
        _CALIBRATION_ROLE: [],
        _FINAL_ROLE: [],
    }
    for record in assignment.get("records", []):
        role = record.get("identity_role")
        if role in role_records:
            role_records[role].append(record)

    result: dict[str, Any] = {}
    identity_sets: dict[str, set[str]] = {}
    for output_role, assignment_role, stage in (
        ("calibration", _CALIBRATION_ROLE, ExposureStage.CALIBRATION_SCORED),
        ("final", _FINAL_ROLE, ExposureStage.FINAL_TEST_SCORED),
    ):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        subject_by_identity: dict[str, str] = {}
        for record in role_records[assignment_role]:
            sample_token = record.get("sample_token")
            label = label_by_sample.get(sample_token)
            if label is None:
                raise ValueError("selected holdout sample lacks evaluator labels")
            if (
                record.get("dataset_name") != DOGFACE_DATASET
                or record.get("source_variant") != "original"
                or label.get("original_split") != "train"
                or label.get("region") != "FACE"
            ):
                raise ValueError(
                    "selected holdout samples must be original DogFace publisher-train face crops"
                )
            identity_token = record.get("identity_token")
            dataset_identity_id = label.get("dataset_identity_id")
            if (
                not isinstance(identity_token, str)
                or not isinstance(dataset_identity_id, str)
                or compute_identity_token(dataset_identity_id) != identity_token
                or label.get("identity_token") != identity_token
            ):
                raise ValueError("selected holdout evaluator identity binding differs")
            subject = compute_public_subject_token(dataset_identity_id)
            previous = subject_by_identity.setdefault(identity_token, subject)
            if previous != subject:
                raise ValueError("selected holdout identity maps to multiple public subjects")
            grouped[identity_token].append(
                {
                    "sample_token": sample_token,
                    "identity_token": identity_token,
                    "public_subject_token": subject,
                    "registered_dog_id": compute_registered_dog_id(dataset_identity_id),
                }
            )

        if len(grouped) != expected_identities:
            raise ValueError(
                f"{assignment_role} must contain exactly {expected_identities} identities"
            )
        if any(len(samples) < 2 for samples in grouped.values()):
            raise ValueError("every selected holdout identity requires at least two samples")

        identities: list[dict[str, Any]] = []
        one_gallery: list[str] = []
        one_queries: list[str] = []
        three_gallery: list[str] = []
        three_queries: list[str] = []
        three_identities = 0
        for identity_token in sorted(grouped):
            samples = sorted(grouped[identity_token], key=lambda item: item["sample_token"])
            tokens = [item["sample_token"] for item in samples]
            one_gallery.append(tokens[0])
            one_queries.extend(tokens[1:])
            three_eligible = len(tokens) >= 4
            if three_eligible:
                three_identities += 1
                three_gallery.extend(tokens[:3])
                three_queries.extend(tokens[3:])
            identities.append(
                {
                    "identity_token": identity_token,
                    "public_subject_token": samples[0]["public_subject_token"],
                    "registered_dog_id": samples[0]["registered_dog_id"],
                    "sample_tokens": tokens,
                    "three_shot_eligible": three_eligible,
                }
            )
        if three_identities == 0:
            raise ValueError("selected holdout has no identity eligible for three-shot evaluation")
        one_gallery.sort()
        one_queries.sort()
        three_gallery.sort()
        three_queries.sort()
        all_samples = sorted(
            sample["sample_token"]
            for samples in grouped.values()
            for sample in samples
        )
        result[output_role] = {
            "assignment_role": assignment_role,
            "exposure_stage": stage.value,
            "identities": identities,
            "all_sample_tokens": all_samples,
            "one_shot": {
                "gallery_sample_tokens": one_gallery,
                "query_sample_tokens": one_queries,
            },
            "three_shot": {
                "gallery_sample_tokens": three_gallery,
                "query_sample_tokens": three_queries,
            },
        }
        identity_sets[output_role] = set(grouped)
    if identity_sets["calibration"] & identity_sets["final"]:
        raise ValueError("calibration and final DogFace identities overlap")
    return result


def _reject_prior_exposure(
    populations: Mapping[str, Any], ledger: RoleExposureLedger
) -> None:
    prior_samples = {record.sample_token for record in ledger.records}
    prior_identities = {record.identity_token for record in ledger.records}
    prior_subjects = {record.public_subject_token for record in ledger.records}
    for role in ("calibration", "final"):
        population = populations[role]
        if prior_samples.intersection(population["all_sample_tokens"]):
            raise ValueError("selected DogFace samples have prior role exposure")
        for identity in population["identities"]:
            if identity["identity_token"] in prior_identities or (
                identity["public_subject_token"] in prior_subjects
            ):
                raise ValueError("selected DogFace identities have prior role exposure")


def _population_summary(population: Mapping[str, Any]) -> dict[str, Any]:
    identities = population["identities"]
    return {
        "assignment_role": population["assignment_role"],
        "exposure_stage": population["exposure_stage"],
        "identity_count": len(identities),
        "sample_count": len(population["all_sample_tokens"]),
        "three_shot_identity_count": sum(
            identity["three_shot_eligible"] for identity in identities
        ),
        "one_shot_gallery_count": len(
            population["one_shot"]["gallery_sample_tokens"]
        ),
        "one_shot_query_count": len(
            population["one_shot"]["query_sample_tokens"]
        ),
        "three_shot_gallery_count": len(
            population["three_shot"]["gallery_sample_tokens"]
        ),
        "three_shot_query_count": len(
            population["three_shot"]["query_sample_tokens"]
        ),
        "population_sha256": content_sha256(population),
    }


def _declaration_for_plan(
    plan_sha256: str, populations: Mapping[str, Any]
) -> RoleExposureDeclaration:
    records: list[RoleExposureDeclarationRecord] = []
    for role in ("calibration", "final"):
        stage = ExposureStage(populations[role]["exposure_stage"])
        for identity in populations[role]["identities"]:
            records.extend(
                RoleExposureDeclarationRecord(
                    sample_token=sample_token,
                    identity_token=identity["identity_token"],
                    public_subject_token=identity["public_subject_token"],
                    stage=stage,
                )
                for sample_token in identity["sample_tokens"]
            )
    return RoleExposureDeclaration(
        source_artifact_sha256=plan_sha256,
        kind=ExposureDeclarationKind.PRIOR_EVALUATION,
        revoked=False,
        records=tuple(
            sorted(
                records,
                key=lambda item: (
                    item.sample_token,
                    item.identity_token,
                    item.public_subject_token,
                ),
            )
        ),
    )


def _build_pair_weights(
    query_identity_ids: Sequence[str], gallery_identity_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(query_identity_ids, dtype=object)
    gallery = np.asarray(gallery_identity_ids, dtype=object)
    if queries.ndim != 1 or gallery.ndim != 1 or len(queries) == 0 or len(gallery) < 2:
        raise ValueError("pair weighting requires non-empty 1-D query and gallery IDs")
    labels = (queries[:, None] == gallery[None, :]).astype(np.int64)
    weights = np.zeros(labels.shape, dtype=np.float64)
    for identity in sorted(set(queries.tolist())):
        identity_rows = queries == identity
        positive_mask = identity_rows[:, None] & (labels == 1)
        negative_mask = identity_rows[:, None] & (labels == 0)
        positives = int(np.sum(positive_mask))
        negatives = int(np.sum(negative_mask))
        if positives == 0 or negatives == 0:
            raise ValueError("each query identity requires positive and negative pairs")
        weights[positive_mask] = 0.5 / positives
        weights[negative_mask] = 0.5 / negatives
    return labels.ravel(), weights.ravel()


def _serialize_isotonic(model: Any) -> dict[str, Any]:
    return {
        "x_thresholds": np.asarray(model.X_thresholds_, dtype=np.float64).tolist(),
        "y_thresholds": np.asarray(model.y_thresholds_, dtype=np.float64).tolist(),
        "out_of_bounds": "clip",
    }


def _fit_oof_isotonic(
    scores: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    *,
    fitter: Callable[[np.ndarray, np.ndarray], Any] = fit_isotonic_calibration,
) -> tuple[np.ndarray, dict[int, Any], list[dict[str, Any]]]:
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(fold_ids, dtype=np.int64)
    if score_values.ndim != 1 or label_values.shape != score_values.shape or (
        folds.shape != score_values.shape
    ):
        raise ValueError("OOF calibration arrays must be aligned 1-D values")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2 or not np.array_equal(unique_folds, np.arange(len(unique_folds))):
        raise ValueError("OOF calibration folds must be contiguous from zero")
    probabilities = np.empty_like(score_values)
    models: dict[int, Any] = {}
    reports: list[dict[str, Any]] = []
    for fold in unique_folds:
        held_out = folds == fold
        training = ~held_out
        if set(np.unique(label_values[training])) != {0, 1} or set(
            np.unique(label_values[held_out])
        ) != {0, 1}:
            raise ValueError("every OOF calibration train and held-out fold needs both classes")
        model = fitter(score_values[training], label_values[training])
        transformed = np.asarray(model.predict(score_values[held_out]), dtype=np.float64)
        if transformed.shape != (int(np.sum(held_out)),) or not np.all(
            np.isfinite(transformed)
        ):
            raise ValueError("OOF isotonic transform returned invalid probabilities")
        probabilities[held_out] = transformed
        models[int(fold)] = model
        reports.append(
            {
                "held_out_fold": int(fold),
                "training_folds": [int(value) for value in unique_folds if value != fold],
                "training_pair_count": int(np.sum(training)),
                "held_out_pair_count": int(np.sum(held_out)),
                "thresholds": _serialize_isotonic(model),
            }
        )
    return probabilities, models, reports


def _select_aggregation(results: Mapping[str, Mapping[str, Any]]) -> str:
    if not results:
        raise ValueError("aggregation selection requires candidate results")
    for candidate_id, metrics in results.items():
        if candidate_id not in {item["id"] for item in _AGGREGATION_CANDIDATES}:
            raise ValueError("aggregation result contains an unfrozen candidate")
        for metric in ("Rank-1", "MRR"):
            value = metrics.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("aggregation result metric is invalid")
    return min(
        results,
        key=lambda candidate_id: (
            -float(results[candidate_id]["Rank-1"]),
            -float(results[candidate_id]["MRR"]),
            candidate_id,
        ),
    )


def _validate_private_report(report: Mapping[str, Any]) -> None:
    stack: list[Any] = [report]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            forbidden = _FORBIDDEN_REPORT_KEYS.intersection(value)
            if forbidden:
                raise ValueError(
                    "private report contains forbidden row-level field(s): "
                    + ", ".join(sorted(forbidden))
                )
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


def _candidate_by_id(candidate_id: str) -> dict[str, Any]:
    for candidate in _AGGREGATION_CANDIDATES:
        if candidate["id"] == candidate_id:
            return candidate
    raise ValueError("selected aggregation is absent from the frozen candidate set")


def _aggregate_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "num_queries",
        "num_gallery_templates",
        "num_gallery_identities",
        "closed_set",
        "ranking_unit",
        "aggregation",
        "aggregation_parameters",
        "tie_policy",
        "self_match_policy",
        "MRR",
        "Rank-1",
        "Rank-5",
        "Rank-10",
    )
    return {name: result[name] for name in fields if name in result}


def _evaluate_matrix(
    scores: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_tokens: Sequence[str],
    gallery_tokens: Sequence[str],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in candidate.items()
        if key in {"aggregation", "top_k", "temperature"}
    }
    return evaluate_multi_template_closed_set(
        scores,
        query_ids,
        gallery_ids,
        self_match_policy="exclude",
        query_template_ids=np.asarray(query_tokens, dtype=object),
        gallery_template_ids=np.asarray(gallery_tokens, dtype=object),
        rank_ks=(1, 5, 10),
        **kwargs,
    )


def _source_and_model_bindings(
    args: argparse.Namespace,
) -> tuple[
    tuple[Any, ...],
    dict[str, Any],
    dict[str, Any],
    frozenset[str],
    frozenset[str],
]:
    source_spec, source_spec_binding = _document_binding(args.source_spec)
    sources = external._source_spec_from_payload(source_spec)
    _, archive_provenance = external._derive_manifest_records(sources)
    source_files: list[dict[str, Any]] = []
    for source, provenance in zip(sources, archive_provenance, strict=True):
        external._verify_file_sha256(source.archive_path, provenance["archive_sha256"])
        archive_binding = {
            "path": os.fspath(source.archive_path),
            "sha256": provenance["archive_sha256"],
            "byte_size": source.archive_path.stat(follow_symlinks=False).st_size,
        }
        source_files.append(
            {
                "dataset_name": source.dataset_name,
                "archive": archive_binding,
                "archive_receipt": _file_binding(source.archive_receipt_path),
                "dogface_classes_train": (
                    _file_binding(source.dogface_classes_train_path)
                    if source.dogface_classes_train_path is not None
                    else None
                ),
                "dogface_classes_test": (
                    _file_binding(source.dogface_classes_test_path)
                    if source.dogface_classes_test_path is not None
                    else None
                ),
                "archive_receipt_sha256": provenance["archive_receipt_sha256"],
            }
        )
    bindings = {
        "source_spec": source_spec_binding,
        "source_files": source_files,
        "model_directory": {
            "path": os.fspath(args.model_dir),
            "frozen_model_sha256": args.frozen_model_sha256,
        },
        "appearance_checkpoint": _file_binding(args.appearance_checkpoint),
        "face_checkpoint": _file_binding(args.face_checkpoint),
        "weight_intake_bundle": _file_binding(args.weight_intake_bundle),
        "preprocessor_intake_bundle": _file_binding(
            args.preprocessor_intake_bundle
        ),
        "dependency_lock": _file_binding(_repository_root() / "uv.lock"),
    }
    model_metadata, appearance_subjects, face_identities = _load_models(
        args, return_models=False
    )
    return (
        sources,
        bindings,
        {
            **model_metadata,
            "appearance_training_subject_set_sha256": content_sha256(
                sorted(appearance_subjects)
            ),
            "face_training_identity_set_sha256": content_sha256(
                sorted(face_identities)
            ),
        },
        appearance_subjects,
        face_identities,
    )


def _load_models(
    args: argparse.Namespace, *, return_models: bool
) -> Any:
    appearance_args = SimpleNamespace(
        checkpoint=args.appearance_checkpoint,
        checkpoint_sha256=args.appearance_checkpoint_sha256,
        model_dir=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        frozen_model_sha256=args.frozen_model_sha256,
        batch_size=getattr(args, "batch_size", 1),
    )
    frozen, appearance_wrapper, appearance_provenance = external._load_models(
        appearance_args
    )
    del frozen
    appearance_bytes = external._read_sha256_pinned_bytes(
        args.appearance_checkpoint, args.appearance_checkpoint_sha256
    )
    appearance_payload = torch.load(
        io.BytesIO(appearance_bytes), map_location="cpu", weights_only=True
    )
    label_to_index = appearance_payload.get("label_to_index")
    if (
        not isinstance(label_to_index, dict)
        or not label_to_index
        or any(not isinstance(value, str) or not value for value in label_to_index)
        or label_to_index
        != {
            label: index for index, label in enumerate(sorted(label_to_index))
        }
    ):
        raise ValueError("Appearance checkpoint training subject index differs")
    appearance_subjects = frozenset(label_to_index)
    del appearance_payload, appearance_bytes

    face_bytes = external._read_sha256_pinned_bytes(
        args.face_checkpoint, args.face_checkpoint_sha256
    )
    face_checkpoint = torch.load(
        io.BytesIO(face_bytes), map_location="cpu", weights_only=True
    )
    repository = _repository_root()
    face_contract = expected_faceid_contract_for_checkpoint(
        face_checkpoint["faceid_contract"],
        repository,
        architecture="regional_v4",
    )
    training_identities = validate_checkpoint_structure(
        face_checkpoint, expected_faceid_contract=face_contract
    )
    face_backbone, dino_contract = load_receipt_bound_frozen_dino(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
    )
    observed_dino = normalize_dino_local_artifact_contract(
        {
            "model_sha256": dino_contract.model_sha256,
            "preprocessor_sha256": dino_contract.preprocessor_sha256,
            "weight_receipt_sha256": dino_contract.weight_receipt_sha256,
            "preprocessor_receipt_sha256": dino_contract.preprocessor_receipt_sha256,
            "config_sha256": dino_contract.config_sha256,
            "weight_source_contract_sha256": (
                dino_contract.weight_source.contract_sha256
            ),
            "preprocessor_source_contract_sha256": (
                dino_contract.preprocessor_source.contract_sha256
            ),
        }
    )
    if observed_dino["model_sha256"] != args.frozen_model_sha256:
        raise ValueError("Face DINO model SHA-256 differs from the frozen model pin")
    validate_checkpoint_runtime_bindings(
        face_checkpoint,
        observed_dino_local_artifact_contract=observed_dino,
        observed_weight_intake_bundle_sha256=external._sha256_file(
            args.weight_intake_bundle
        ),
        observed_preprocessor_intake_bundle_sha256=external._sha256_file(
            args.preprocessor_intake_bundle
        ),
    )
    face_model = build_faceid_model(face_backbone, dino_contract)
    face_model.encoder.load_state_dict(
        face_checkpoint["encoder_state_dict"], strict=True
    )
    face_model.quality_head.load_state_dict(
        face_checkpoint["quality_head_state_dict"], strict=True
    )
    metadata = {
        "appearance_model_version": "Appearance-v3",
        "appearance_checkpoint_schema": "cvi.training_checkpoint.v1",
        "appearance_checkpoint_sha256": args.appearance_checkpoint_sha256,
        "appearance_training_admission_receipt_sha256": appearance_provenance[
            "trained_checkpoint_training_admission_receipt_sha256"
        ],
        "face_model_version": "Face-v4",
        "face_checkpoint_schema": face_checkpoint["schema_version"],
        "face_checkpoint_sha256": args.face_checkpoint_sha256,
        "faceid_contract_sha256": face_checkpoint["faceid_contract_sha256"],
        "dino_local_artifact_contract_sha256": face_checkpoint[
            "dino_local_artifact_contract_sha256"
        ],
        "frozen_model_sha256": args.frozen_model_sha256,
        "face_training_identity_count": len(training_identities),
    }
    del face_checkpoint, face_bytes
    if not return_models:
        del appearance_wrapper, face_model
        return metadata, appearance_subjects, frozenset(training_identities)
    # The trained Appearance wrapper normalizes in forward; holdout extraction
    # uses the exact checkpoint-loaded HF backbone to retain pre-L2 CLS vectors.
    appearance_backbone = appearance_wrapper._backbone
    return (
        appearance_backbone,
        face_model,
        appearance_subjects,
        frozenset(training_identities),
        metadata,
    )


def _validate_training_disjointness(
    populations: Mapping[str, Any],
    appearance_training_subjects: set[str] | frozenset[str],
    face_training_identities: set[str] | frozenset[str],
) -> None:
    selected_identities = {
        identity["registered_dog_id"]
        for role in ("calibration", "final")
        for identity in populations[role]["identities"]
    }
    selected_subjects = {
        identity["public_subject_token"]
        for role in ("calibration", "final")
        for identity in populations[role]["identities"]
    }
    if selected_subjects & appearance_training_subjects:
        raise ValueError("Appearance checkpoint training subjects overlap the holdout")
    if selected_identities & face_training_identities:
        raise ValueError("Face checkpoint training identities overlap the holdout")


def _make_plan(
    *,
    args: argparse.Namespace,
    populations: Mapping[str, Any],
    bindings: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "PASS_DOGFACE_HOLDOUT_PREPARED",
        "interpretation": (
            "DOGFACE_DEVELOPMENT_IS_CALIBRATION_AND_DOGFACE_CALIBRATION_IS_"
            "ADVANCED_ONCE_TO_FINAL_TEST_SCORED_BEFORE_ANY_SCORE_COMPUTATION"
        ),
        "input_bindings": dict(bindings),
        "model_bindings": dict(model_metadata),
        "populations": dict(populations),
        "population_summaries": {
            role: _population_summary(populations[role])
            for role in ("calibration", "final")
        },
        "protocol": {
            "folds": _FOLDS,
            "fold_seed": _OOF_SEED,
            "fold_unit": "query_identity",
            "fold_assignment": "build_dataset_balanced_oof_protocol",
            "calibration": "isotonic_per_channel",
            "fusion": {
                "method": "fit_oof_simplex",
                "config": asdict(OOFSimplexConfig()),
                "pair_weighting": (
                    "equal_total_per_query_identity_and_half_positive_half_negative"
                ),
                "channels": ["Appearance-v3", "Face-v4"],
            },
            "aggregation_candidates": [dict(item) for item in _AGGREGATION_CANDIDATES],
            "aggregation_selection": (
                "highest_Rank-1_then_highest_MRR_then_lexical_candidate_id"
            ),
            "tie_policy": "stable_first_gallery_identity_occurrence",
            "one_shot": "first_token_sorted_sample_gallery_remaining_queries",
            "three_shot": (
                "first_three_token_sorted_samples_gallery_remaining_queries_for_"
                "identities_with_at_least_four_samples"
            ),
        },
        "code_sha256s": _code_hashes(),
        "output_contract": {
            "prepare_directory": os.fspath(args.output_dir),
            "files": [
                "plan.json",
                "exposure_declaration.json",
                "exposure_ledger.json",
                "exposure_receipt.json",
            ],
            "private_mode": "0600_files_in_0700_directory",
            "no_replace": True,
        },
    }


def _prepare(args: argparse.Namespace) -> None:
    _require_absolute_inputs(
        args,
        (
            "assignment",
            "labels",
            "source_bundle",
            "split_receipt",
            "source_spec",
            "historical_exposure_ledger",
            "historical_exposure_receipt",
            "appearance_checkpoint",
            "face_checkpoint",
            "model_dir",
            "weight_intake_bundle",
            "preprocessor_intake_bundle",
            "output_dir",
        ),
    )
    _refuse_existing((args.output_dir,))
    assignment, labels, source, _, document_bindings = _load_authenticated_split(args)
    historical_ledger, historical_receipt, exposure_bindings = _load_exposure_history(
        args.historical_exposure_ledger,
        args.historical_exposure_receipt,
        args.historical_exposure_receipt_sha256,
        source,
    )
    split_receipt = read_strict_json_document(args.split_receipt).payload
    if split_receipt.get("role_exposure_ledger_sha256") != (
        historical_ledger.ledger_sha256
    ) or split_receipt.get("role_exposure_receipt_sha256") != (
        historical_receipt.receipt_sha256
    ):
        raise ValueError("protected split receipt does not bind the exposure history")
    populations = _build_holdout_populations(assignment, labels)
    _reject_prior_exposure(populations, historical_ledger)
    (
        _,
        source_model_bindings,
        model_metadata,
        appearance_training_subjects,
        face_training_identities,
    ) = _source_and_model_bindings(args)
    _validate_training_disjointness(
        populations, appearance_training_subjects, face_training_identities
    )
    plan = _make_plan(
        args=args,
        populations=populations,
        bindings={
            **document_bindings,
            **exposure_bindings,
            **source_model_bindings,
            "expected_pins": {
                "split_receipt_sha256": args.split_receipt_sha256,
                "historical_exposure_receipt_sha256": (
                    args.historical_exposure_receipt_sha256
                ),
                "appearance_checkpoint_sha256": args.appearance_checkpoint_sha256,
                "face_checkpoint_sha256": args.face_checkpoint_sha256,
                "frozen_model_sha256": args.frozen_model_sha256,
            },
        },
        model_metadata=model_metadata,
    )
    plan_sha256 = content_sha256(plan)
    declaration = _declaration_for_plan(plan_sha256, populations)
    merged = merge_role_exposure_declarations(
        (*historical_ledger.declarations, declaration)
    )
    receipt = create_role_exposure_receipt(merged)
    verify_role_exposure_receipt(merged, receipt)
    write_private_json_directory_bundle(
        args.output_dir,
        (
            ("plan.json", plan),
            ("exposure_declaration.json", declaration.to_dict()),
            ("exposure_ledger.json", merged.to_dict()),
            ("exposure_receipt.json", receipt.to_dict()),
        ),
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "output_dir": os.fspath(args.output_dir),
                "plan_sha256": plan_sha256,
                "exposure_receipt_sha256": receipt.receipt_sha256,
                "calibration": plan["population_summaries"]["calibration"],
                "final": plan["population_summaries"]["final"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _verify_plan_and_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any], tuple[Any, ...]]:
    plan_document = read_strict_json_document(args.plan)
    plan = plan_document.payload
    if plan_document.canonical_payload_sha256 != args.plan_sha256:
        raise ValueError("plan content SHA-256 differs from the external pin")
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != (
        "PASS_DOGFACE_HOLDOUT_PREPARED"
    ):
        raise ValueError("DogFace holdout plan schema or status differs")
    _verify_code_bindings(plan)
    assignment, labels, source, source_by_token, document_bindings = (
        _load_authenticated_split(args)
    )
    historical_ledger, _, exposure_bindings = _load_exposure_history(
        args.historical_exposure_ledger,
        args.historical_exposure_receipt,
        args.historical_exposure_receipt_sha256,
        source,
    )
    (
        sources,
        source_model_bindings,
        model_metadata,
        appearance_training_subjects,
        face_training_identities,
    ) = _source_and_model_bindings(args)
    observed_bindings = {
        **document_bindings,
        **exposure_bindings,
        **source_model_bindings,
        "expected_pins": {
            "split_receipt_sha256": args.split_receipt_sha256,
            "historical_exposure_receipt_sha256": (
                args.historical_exposure_receipt_sha256
            ),
            "appearance_checkpoint_sha256": args.appearance_checkpoint_sha256,
            "face_checkpoint_sha256": args.face_checkpoint_sha256,
            "frozen_model_sha256": args.frozen_model_sha256,
        },
    }
    if plan.get("input_bindings") != observed_bindings:
        raise ValueError("one or more prepared input bindings differ")
    if plan.get("model_bindings") != model_metadata:
        raise ValueError("prepared model bindings differ")
    populations = _build_holdout_populations(assignment, labels)
    if plan.get("populations") != populations:
        raise ValueError("prepared DogFace holdout populations differ")
    _reject_prior_exposure(populations, historical_ledger)
    _validate_training_disjointness(
        populations, appearance_training_subjects, face_training_identities
    )

    merged_payload = read_strict_json_document(args.exposure_ledger).payload
    merged_receipt_payload = read_strict_json_document(args.exposure_receipt).payload
    merged = RoleExposureLedger.from_dict(merged_payload)
    merged_receipt = RoleExposureReceipt.from_dict(merged_receipt_payload)
    verify_role_exposure_receipt(merged, merged_receipt)
    if merged_receipt.receipt_sha256 != args.exposure_receipt_sha256:
        raise ValueError("merged exposure receipt differs from the external pin")
    expected_declaration = _declaration_for_plan(args.plan_sha256, populations)
    expected_merged = merge_role_exposure_declarations(
        (*historical_ledger.declarations, expected_declaration)
    )
    if merged != expected_merged or merged_receipt != create_role_exposure_receipt(
        expected_merged
    ):
        raise ValueError("merged exposure ledger does not exactly bind the plan")
    verify_split_role_exposure_inputs(source.samples, merged, merged_receipt)
    return plan, populations, source, source_by_token, sources


def _identity_by_sample(population: Mapping[str, Any]) -> dict[str, str]:
    return {
        sample_token: identity["identity_token"]
        for identity in population["identities"]
        for sample_token in identity["sample_tokens"]
    }


def _population_inputs(
    population: Mapping[str, Any], shot: str
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    token_to_identity = _identity_by_sample(population)
    gallery = population[shot]["gallery_sample_tokens"]
    queries = population[shot]["query_sample_tokens"]
    return (
        gallery,
        queries,
        np.asarray([token_to_identity[token] for token in gallery], dtype=object),
        np.asarray([token_to_identity[token] for token in queries], dtype=object),
    )


def _extract_holdout_channels(
    *,
    args: argparse.Namespace,
    populations: Mapping[str, Any],
    source_by_token: Mapping[str, Any],
    sources: tuple[Any, ...],
    appearance_backbone: torch.nn.Module,
    face_model: torch.nn.Module,
    roles: Sequence[str],
) -> dict[str, Any]:
    if not roles or any(role not in {"calibration", "final"} for role in roles):
        raise ValueError("channel extraction roles must contain calibration or final")
    if len(set(roles)) != len(roles):
        raise ValueError("channel extraction roles must be unique")
    all_tokens = sorted(
        {
            token
            for role in roles
            for token in populations[role]["all_sample_tokens"]
        }
    )
    members = tuple(
        external.PopulationMember(
            sample_token=token,
            identity_token=source_by_token[token].identity_token,
            event_token=token,
            bootstrap_cluster_token=None,
        )
        for token in all_tokens
    )
    binding_population = external.Population(
        key=external.PopulationKey("DOGFACE_HOLDOUT_FUSION", "BOUND", 0, 1),
        gallery=members,
        queries=(),
    )
    manifest_records, archive_provenance = external._derive_manifest_records(sources)
    locations = external._bind_image_locations(
        (binding_population,), source_by_token, manifest_records, sources
    )
    dogface_source = next(
        source for source in sources if source.dataset_name == DOGFACE_DATASET
    )
    dogface_provenance = next(
        item for item in archive_provenance if item["dataset_name"] == DOGFACE_DATASET
    )
    device = torch.device(args.device)
    appearance_backbone.to(device).eval()
    face_model.to(device).eval()
    appearance: dict[str, np.ndarray] = {}
    face: dict[str, np.ndarray] = {}
    face_quality: dict[str, float] = {}
    dino_baseline: dict[str, np.ndarray] = {}
    regional_projection: dict[str, np.ndarray] = {}
    captured: dict[str, torch.Tensor] = {}

    def capture_dino_output(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        baseline = getattr(output, "pooler_output", None)
        if not isinstance(baseline, torch.Tensor):
            raise RuntimeError("Face DINO pooler output is unavailable")
        captured["dino_baseline"] = baseline.float()

    def capture_regional_projection(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Face regional projection output is unavailable")
        captured["regional_projection"] = output.float()

    hooks = (
        face_model.dino.register_forward_hook(capture_dino_output),
        face_model.encoder.projection.register_forward_hook(
            capture_regional_projection
        ),
    )
    try:
        with ExitStack() as stack:
            archive = external._open_verified_archive(
                stack, dogface_source.archive_path, dogface_provenance["archive_sha256"]
            )
            for offset in range(0, len(all_tokens), args.batch_size):
                tokens = all_tokens[offset : offset + args.batch_size]
                images = [
                    external._decode_image(archive, locations[token].record)
                    for token in tokens
                ]
                if any(image.size != (224, 224) for image in images):
                    raise ValueError("selected DogFace crops must already be 224x224")
                arrays = [np.asarray(image, dtype=np.uint8) for image in images]
                raw = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2)).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                raw.div_(255.0)
                mean = torch.tensor(
                    external._PREPROCESSING["mean"], device=device
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    external._PREPROCESSING["std"], device=device
                ).view(1, 3, 1, 1)
                normalized = (raw - mean) / std
                captured.clear()
                with torch.inference_mode():
                    appearance_output = appearance_backbone(pixel_values=normalized)
                    hidden = getattr(appearance_output, "last_hidden_state", None)
                    if not isinstance(hidden, torch.Tensor) or hidden.shape != (
                        len(tokens),
                        257,
                        384,
                    ):
                        raise RuntimeError(
                            "Appearance backbone must return [B,257,384] hidden states"
                        )
                    appearance_cls = hidden[:, 0].float()
                    with torch.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        face_output = face_model(raw, landmarks=None)
                    face_embedding = face_output.get("embedding")
                    quality = face_output.get("quality")
                    baseline = captured.get("dino_baseline")
                    regional = captured.get("regional_projection")
                    if (
                        not isinstance(face_embedding, torch.Tensor)
                        or face_embedding.shape != (len(tokens), 640)
                        or not isinstance(quality, torch.Tensor)
                        or quality.shape != (len(tokens),)
                        or not isinstance(baseline, torch.Tensor)
                        or baseline.shape != (len(tokens), 384)
                        or not isinstance(regional, torch.Tensor)
                        or regional.shape != (len(tokens), 256)
                    ):
                        raise RuntimeError("Face model or diagnostic hook contract differs")
                tensors = (appearance_cls, face_embedding, quality, baseline, regional)
                if any(not torch.isfinite(value).all() for value in tensors):
                    raise RuntimeError("holdout model output contains non-finite values")
                values = [value.detach().cpu().numpy() for value in tensors]
                for index, token in enumerate(tokens):
                    appearance[token] = values[0][index].astype(np.float32, copy=False)
                    face[token] = values[1][index].astype(np.float32, copy=False)
                    face_quality[token] = float(values[2][index])
                    dino_baseline[token] = values[3][index].astype(
                        np.float32, copy=False
                    )
                    regional_projection[token] = values[4][index].astype(
                        np.float32, copy=False
                    )
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "appearance": appearance,
        "face": face,
        "face_quality": face_quality,
        "dino_baseline": dino_baseline,
        "regional_projection": regional_projection,
        "regional_projection_status": (
            "AVAILABLE_PRE_NORMALIZATION_FROM_BOUND_FORWARD_HOOK"
        ),
    }


def _channel_matrix(
    cache: Mapping[str, np.ndarray], gallery: Sequence[str], queries: Sequence[str]
) -> np.ndarray:
    return compute_cosine_score_matrix(
        np.stack([cache[token] for token in queries]),
        np.stack([cache[token] for token in gallery]),
    )


def _face_pair_quality(
    quality: Mapping[str, float], gallery: Sequence[str], queries: Sequence[str]
) -> np.ndarray:
    query_quality = np.asarray([quality[token] for token in queries], dtype=np.float64)
    gallery_quality = np.asarray(
        [quality[token] for token in gallery], dtype=np.float64
    )
    return np.sqrt(query_quality[:, None] * gallery_quality[None, :])


def _fuse_probabilities(
    model: Any,
    appearance: np.ndarray,
    face: np.ndarray,
    face_quality: np.ndarray,
) -> np.ndarray:
    shape = appearance.shape
    values = np.column_stack((appearance.ravel(), face.ravel()))
    quality = np.column_stack(
        (np.ones(values.shape[0], dtype=np.float64), face_quality.ravel())
    )
    availability = np.ones(values.shape, dtype=bool)
    return model.predict_proba(
        values, availability=availability, quality=quality
    ).reshape(shape)


def _fold_transform_matrix(
    matrix: np.ndarray,
    query_fold_ids: np.ndarray,
    models: Mapping[int, Any],
) -> np.ndarray:
    transformed = np.empty(matrix.shape, dtype=np.float64)
    for fold, model in models.items():
        rows = query_fold_ids == fold
        transformed[rows] = np.asarray(
            model.predict(matrix[rows].ravel()), dtype=np.float64
        ).reshape((int(np.sum(rows)), matrix.shape[1]))
    if not np.all(np.isfinite(transformed)):
        raise ValueError("fold-specific isotonic transform produced non-finite values")
    return transformed


def _identity_bootstrap(
    result: Mapping[str, Any], *, resamples: int, seed: int
) -> dict[str, Any]:
    return {
        "Rank-1": identity_clustered_bootstrap_ci(
            result["query_rows"],
            metric="Rank-1",
            resamples=resamples,
            seed=seed,
        ),
        "MRR": identity_clustered_bootstrap_ci(
            result["query_rows"],
            metric="reciprocal_rank",
            resamples=resamples,
            seed=seed + 1,
        ),
    }


def _diagnostics(
    populations: Mapping[str, Any], channels: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("calibration", "final"):
        population = populations[role]
        tokens = population["all_sample_tokens"]
        token_to_identity = _identity_by_sample(population)
        identities = np.asarray([token_to_identity[token] for token in tokens], dtype=object)
        qualities = np.asarray(
            [channels["face_quality"][token] for token in tokens], dtype=np.float64
        )
        result[role] = {
            "Appearance-v3-pre-L2-CLS": compute_embedding_diagnostics(
                np.stack([channels["appearance"][token] for token in tokens]),
                identity_ids=identities,
                domain_ids=np.asarray([DOGFACE_DATASET] * len(tokens), dtype=object),
            ),
            "Face-v4-final-640D": compute_embedding_diagnostics(
                np.stack([channels["face"][token] for token in tokens]),
                identity_ids=identities,
                domain_ids=np.asarray([DOGFACE_DATASET] * len(tokens), dtype=object),
                quality_scores=qualities,
            ),
            "Face-v4-DINO-baseline-384D": compute_embedding_diagnostics(
                np.stack([channels["dino_baseline"][token] for token in tokens]),
                identity_ids=identities,
                domain_ids=np.asarray([DOGFACE_DATASET] * len(tokens), dtype=object),
                quality_scores=qualities,
            ),
            "Face-v4-regional-projection-256D": compute_embedding_diagnostics(
                np.stack(
                    [channels["regional_projection"][token] for token in tokens]
                ),
                identity_ids=identities,
                domain_ids=np.asarray([DOGFACE_DATASET] * len(tokens), dtype=object),
                quality_scores=qualities,
            ),
            "face_quality": {
                "count": len(qualities),
                "minimum": float(np.min(qualities)),
                "median": float(np.median(qualities)),
                "maximum": float(np.max(qualities)),
                "mean": float(np.mean(qualities)),
            },
            "session_reason": "verified session IDs are unavailable for DogFace",
            "repeat_reason": "repeat IDs are unavailable for DogFace",
        }
    result["regional_projection_status"] = channels["regional_projection_status"]
    return result


def _calibrate_and_select(
    plan: Mapping[str, Any],
    populations: Mapping[str, Any],
    channels: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    population = populations["calibration"]
    gallery, queries, gallery_ids, query_ids = _population_inputs(
        population, "one_shot"
    )
    appearance_raw = _channel_matrix(channels["appearance"], gallery, queries)
    face_raw = _channel_matrix(channels["face"], gallery, queries)
    fold_protocol = build_dataset_balanced_oof_protocol(
        np.asarray([DOGFACE_DATASET] * len(query_ids), dtype=object),
        query_ids,
        config=RobustnessProtocolConfig(
            n_splits=plan["protocol"]["folds"],
            seed=plan["protocol"]["fold_seed"],
            target_exposure_stage=ExposureStage.CALIBRATION_SCORED,
        ),
        historical_exposure_stages=np.asarray(
            [ExposureStage.CALIBRATION_SCORED] * len(query_ids), dtype=object
        ),
    )
    labels, pair_weights = _build_pair_weights(query_ids, gallery_ids)
    pair_folds = np.repeat(fold_protocol.fold_ids, len(gallery_ids))
    appearance_oof, appearance_models, appearance_fold_reports = (
        _fit_oof_isotonic(appearance_raw.ravel(), labels, pair_folds)
    )
    face_oof, face_models, face_fold_reports = _fit_oof_isotonic(
        face_raw.ravel(), labels, pair_folds
    )
    appearance_oof_matrix = appearance_oof.reshape(appearance_raw.shape)
    face_oof_matrix = face_oof.reshape(face_raw.shape)
    face_quality = _face_pair_quality(channels["face_quality"], gallery, queries)
    calibrated_pairs = np.column_stack((appearance_oof, face_oof))
    simplex_quality = np.column_stack(
        (np.ones(len(labels), dtype=np.float64), face_quality.ravel())
    )
    simplex = fit_oof_simplex(
        ("Appearance-v3", "Face-v4"),
        calibrated_pairs,
        labels,
        pair_folds,
        availability=np.ones(calibrated_pairs.shape, dtype=bool),
        quality=simplex_quality,
        sample_weights=pair_weights,
        config=OOFSimplexConfig(**plan["protocol"]["fusion"]["config"]),
    )
    fused_oof = _fuse_probabilities(
        simplex, appearance_oof_matrix, face_oof_matrix, face_quality
    )
    one_candidate = _candidate_by_id("max")
    one_result = _evaluate_matrix(
        fused_oof, query_ids, gallery_ids, queries, gallery, one_candidate
    )
    appearance_one = _evaluate_matrix(
        appearance_oof_matrix,
        query_ids,
        gallery_ids,
        queries,
        gallery,
        one_candidate,
    )
    face_one = _evaluate_matrix(
        face_oof_matrix, query_ids, gallery_ids, queries, gallery, one_candidate
    )

    three_gallery, three_queries, three_gallery_ids, three_query_ids = (
        _population_inputs(population, "three_shot")
    )
    fold_by_identity: dict[str, int] = {}
    for identity, fold in zip(query_ids.tolist(), fold_protocol.fold_ids.tolist(), strict=True):
        if fold_by_identity.setdefault(identity, int(fold)) != int(fold):
            raise RuntimeError("query identity crosses prepared OOF folds")
    three_folds = np.asarray(
        [fold_by_identity[identity] for identity in three_query_ids], dtype=np.int64
    )
    appearance_three_raw = _channel_matrix(
        channels["appearance"], three_gallery, three_queries
    )
    face_three_raw = _channel_matrix(channels["face"], three_gallery, three_queries)
    appearance_three = _fold_transform_matrix(
        appearance_three_raw, three_folds, appearance_models
    )
    face_three = _fold_transform_matrix(face_three_raw, three_folds, face_models)
    face_three_quality = _face_pair_quality(
        channels["face_quality"], three_gallery, three_queries
    )
    fused_three = _fuse_probabilities(
        simplex, appearance_three, face_three, face_three_quality
    )
    candidate_aggregate: dict[str, dict[str, Any]] = {}
    for candidate in plan["protocol"]["aggregation_candidates"]:
        evaluated = _evaluate_matrix(
            fused_three,
            three_query_ids,
            three_gallery_ids,
            three_queries,
            three_gallery,
            candidate,
        )
        candidate_aggregate[candidate["id"]] = _aggregate_metrics(evaluated)
    selected_id = _select_aggregation(candidate_aggregate)

    appearance_full = fit_isotonic_calibration(appearance_raw.ravel(), labels)
    face_full = fit_isotonic_calibration(face_raw.ravel(), labels)
    calibration_report = {
        "fold_protocol": fold_protocol.report,
        "source_assignment_exposure_stage": ExposureStage.MODEL_SELECTION_SCORED.value,
        "evaluation_exposure_stage": ExposureStage.CALIBRATION_SCORED.value,
        "pair_count": len(labels),
        "positive_pair_count": int(np.sum(labels)),
        "negative_pair_count": int(np.sum(labels == 0)),
        "pair_weighting": {
            "identity_total_weight": 1.0,
            "positive_fraction_per_identity": 0.5,
            "negative_fraction_per_identity": 0.5,
        },
        "isotonic": {
            "Appearance-v3": {
                "oof_folds": appearance_fold_reports,
                "full_calibration_thresholds": _serialize_isotonic(appearance_full),
            },
            "Face-v4": {
                "oof_folds": face_fold_reports,
                "full_calibration_thresholds": _serialize_isotonic(face_full),
            },
        },
        "simplex": simplex.report,
        "one_shot_oof": {
            "fused": _aggregate_metrics(one_result),
            "Appearance-v3": _aggregate_metrics(appearance_one),
            "Face-v4": _aggregate_metrics(face_one),
            "probability_calibration": compute_probability_calibration_metrics(
                fused_oof.ravel(), labels
            ),
        },
        "three_shot_oof_candidates": candidate_aggregate,
    }
    frozen = {
        "selected_aggregation_id": selected_id,
        "selected_aggregation": _candidate_by_id(selected_id),
        "selection_order": (
            "highest_Rank-1_then_highest_MRR_then_lexical_candidate_id"
        ),
        "simplex_weights": simplex.weights.tolist(),
        "simplex_model": simplex,
        "appearance_calibrator": appearance_full,
        "face_calibrator": face_full,
    }
    return calibration_report, frozen


def _evaluate_final(
    *,
    args: argparse.Namespace,
    population: Mapping[str, Any],
    channels: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    selected = frozen["selected_aggregation"]
    simplex = frozen["simplex_model"]
    appearance_calibrator = frozen["appearance_calibrator"]
    face_calibrator = frozen["face_calibrator"]
    results: dict[str, Any] = {}
    for shot_index, shot in enumerate(("one_shot", "three_shot")):
        gallery, queries, gallery_ids, query_ids = _population_inputs(population, shot)
        appearance_raw = _channel_matrix(channels["appearance"], gallery, queries)
        face_raw = _channel_matrix(channels["face"], gallery, queries)
        appearance = np.asarray(
            appearance_calibrator.predict(appearance_raw.ravel()), dtype=np.float64
        ).reshape(appearance_raw.shape)
        face = np.asarray(
            face_calibrator.predict(face_raw.ravel()), dtype=np.float64
        ).reshape(face_raw.shape)
        quality = _face_pair_quality(channels["face_quality"], gallery, queries)
        fused = _fuse_probabilities(simplex, appearance, face, quality)
        aggregation = _candidate_by_id("max") if shot == "one_shot" else selected
        fused_result = _evaluate_matrix(
            fused, query_ids, gallery_ids, queries, gallery, aggregation
        )
        appearance_max = _evaluate_matrix(
            appearance,
            query_ids,
            gallery_ids,
            queries,
            gallery,
            _candidate_by_id("max"),
        )
        face_max = _evaluate_matrix(
            face,
            query_ids,
            gallery_ids,
            queries,
            gallery,
            _candidate_by_id("max"),
        )
        context: dict[str, Any] = {
            "Appearance-v3-max": _aggregate_metrics(appearance_max),
            "Face-v4-max": _aggregate_metrics(face_max),
        }
        if shot == "three_shot" and selected["id"] != "max":
            context["Appearance-v3-selected-robust"] = _aggregate_metrics(
                _evaluate_matrix(
                    appearance,
                    query_ids,
                    gallery_ids,
                    queries,
                    gallery,
                    selected,
                )
            )
            context["Face-v4-selected-robust"] = _aggregate_metrics(
                _evaluate_matrix(
                    face,
                    query_ids,
                    gallery_ids,
                    queries,
                    gallery,
                    selected,
                )
            )
        results[shot] = {
            "fused": _aggregate_metrics(fused_result),
            "fused_identity_clustered_ci": _identity_bootstrap(
                fused_result,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 10 * shot_index,
            ),
            "channel_context": context,
        }
    return results


def _evaluate(args: argparse.Namespace) -> None:
    _require_absolute_inputs(
        args,
        (
            "plan",
            "assignment",
            "labels",
            "source_bundle",
            "split_receipt",
            "source_spec",
            "historical_exposure_ledger",
            "historical_exposure_receipt",
            "exposure_ledger",
            "exposure_receipt",
            "appearance_checkpoint",
            "face_checkpoint",
            "model_dir",
            "weight_intake_bundle",
            "preprocessor_intake_bundle",
            "output",
        ),
    )
    _refuse_existing((args.output,))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    plan, populations, _, source_by_token, sources = _verify_plan_and_inputs(args)
    (
        appearance,
        face,
        appearance_training_subjects,
        face_training_identities,
        model_metadata,
    ) = _load_models(args, return_models=True)
    _validate_training_disjointness(
        populations, appearance_training_subjects, face_training_identities
    )
    calibration_channels = _extract_holdout_channels(
        args=args,
        populations=populations,
        source_by_token=source_by_token,
        sources=sources,
        appearance_backbone=appearance,
        face_model=face,
        roles=("calibration",),
    )

    # Calibration selection is complete before final model inference starts.
    calibration_report, frozen_runtime = _calibrate_and_select(
        plan, populations, calibration_channels
    )
    frozen_report = {
        key: value
        for key, value in frozen_runtime.items()
        if key
        not in {
            "simplex_model",
            "appearance_calibrator",
            "face_calibrator",
        }
    }
    final_channels = _extract_holdout_channels(
        args=args,
        populations=populations,
        source_by_token=source_by_token,
        sources=sources,
        appearance_backbone=appearance,
        face_model=face,
        roles=("final",),
    )
    final_results = _evaluate_final(
        args=args,
        population=populations["final"],
        channels=final_channels,
        frozen=frozen_runtime,
    )
    channels = {
        name: {**calibration_channels[name], **final_channels[name]}
        for name in (
            "appearance",
            "face",
            "face_quality",
            "dino_baseline",
            "regional_projection",
        )
    }
    channels["regional_projection_status"] = final_channels[
        "regional_projection_status"
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_DOGFACE_HOLDOUT_FUSION_EVALUATION",
        "interpretation": (
            "TWO_STAGE_DOGFACE_CALIBRATION_AND_ONE_TIME_FINAL_CLOSED_SET_EVALUATION"
        ),
        "calibration_results": calibration_report,
        "frozen_selection": frozen_report,
        "final_results": final_results,
        "embedding_diagnostics": _diagnostics(populations, channels),
        "provenance": {
            "plan_sha256": args.plan_sha256,
            "split_receipt_sha256": args.split_receipt_sha256,
            "exposure_receipt_sha256": args.exposure_receipt_sha256,
            "appearance_checkpoint_sha256": args.appearance_checkpoint_sha256,
            "face_checkpoint_sha256": args.face_checkpoint_sha256,
            "frozen_model_sha256": args.frozen_model_sha256,
            "source_spec_sha256": plan["input_bindings"]["source_spec"][
                "content_sha256"
            ],
            "code_sha256s": plan["code_sha256s"],
            "model_bindings": model_metadata,
            "device": args.device,
            "batch_size": args.batch_size,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "limitations": [
            "public DogFace publisher-train-lane holdout, not a private deployment population",
            "verified capture sessions are unavailable",
            "fusion includes Appearance and Face channels only",
            "closed-set holdout performance is not lifelong biometric validation",
        ],
        "privacy": {
            "per_sample_ids_serialized": False,
            "per_sample_vectors_serialized": False,
            "per_pair_values_serialized": False,
            "query_rows_serialized": False,
        },
    }
    _validate_private_report(report)
    write_private_json_bundle(((args.output, report),))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": os.fspath(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _add_split_model_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument(
        "--split-receipt-sha256", required=True, type=_parse_sha256
    )
    parser.add_argument("--source-spec", required=True, type=Path)
    parser.add_argument("--historical-exposure-ledger", required=True, type=Path)
    parser.add_argument("--historical-exposure-receipt", required=True, type=Path)
    parser.add_argument(
        "--historical-exposure-receipt-sha256",
        required=True,
        type=_parse_sha256,
    )
    parser.add_argument("--appearance-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--appearance-checkpoint-sha256", required=True, type=_parse_sha256
    )
    parser.add_argument("--face-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--face-checkpoint-sha256", required=True, type=_parse_sha256
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--weight-intake-bundle", required=True, type=Path)
    parser.add_argument("--preprocessor-intake-bundle", required=True, type=Path)
    parser.add_argument(
        "--frozen-model-sha256", required=True, type=_parse_sha256
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="freeze populations and advance exposure before scores"
    )
    _add_split_model_inputs(prepare)
    prepare.add_argument("--output-dir", required=True, type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="run the exact prepared calibration/final protocol"
    )
    _add_split_model_inputs(evaluate)
    evaluate.add_argument("--plan", required=True, type=Path)
    evaluate.add_argument("--plan-sha256", required=True, type=_parse_sha256)
    evaluate.add_argument("--exposure-ledger", required=True, type=Path)
    evaluate.add_argument("--exposure-receipt", required=True, type=Path)
    evaluate.add_argument(
        "--exposure-receipt-sha256", required=True, type=_parse_sha256
    )
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--bootstrap-seed", type=int, default=0)
    evaluate.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        if args.bootstrap_seed < 0:
            parser.error("--bootstrap-seed must be non-negative")
        if args.bootstrap_resamples <= 0:
            parser.error("--bootstrap-resamples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "prepare":
        _prepare(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
