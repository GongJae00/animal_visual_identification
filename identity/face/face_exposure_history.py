"""Project authenticated face participation into monotonic role exposure history."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from foundation.provenance import content_sha256
from identity.research.dataset_stratified_kfold import (
    DatasetStratifiedIdentityKFoldManifest,
    HeldOutSampleRole,
)
from identity.face.face_public_source_binding import (
    validate_face_public_source_binding_bundle,
)
from identity.exposure.role_exposure import (
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

HISTORY_SCHEMA = "cvi.face_exposure_history.v1"
UNRESOLVED_SCHEMA = "cvi.face_exposure_unresolved_row.v1"
BUNDLE_SCHEMA = "cvi.face_exposure_history_bundle.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INTERPRETATION = (
    "EXACT_DECLARED_PROJECTIONS_ONLY;NO_DECLARED_EXPOSURE_IS_NOT_PROOF_OF_CLEAN_"
    "HISTORY;UNRESOLVED_ROWS_BLOCK_ROLE_ALLOCATION"
)


def build_face_exposure_history(
    token_bridge_bundle: object,
    *,
    full128_artifacts: Sequence[Mapping[str, Any]] = (),
    masked_afn_runs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]] = (),
) -> dict[str, Any]:
    """Build exact declarations and retain every unsupported projection as a blocker."""

    bridge = validate_face_public_source_binding_bundle(token_bridge_bundle)
    bridge_records = bridge["binding"]["records"]
    by_route = {row["route_sample_token"]: row for row in bridge_records}
    by_public = {row["public_sample_token"]: row for row in bridge_records}
    if not full128_artifacts and not masked_afn_runs:
        raise ValueError("at least one Full128 or Masked-AFN artifact is required")

    declarations: list[RoleExposureDeclaration] = []
    unresolved: list[dict[str, Any]] = []
    source_bindings: list[dict[str, Any]] = []
    for artifact in full128_artifacts:
        source_hash = content_sha256(artifact)
        schema = artifact.get("schema_version")
        if schema == "cvi.full128_variant_run.v1":
            records, failures = _project_full128_variant(
                artifact, source_hash=source_hash, by_route=by_route
            )
            kind = "FULL128_VARIANT_RUN"
        else:
            records = ()
            failures = _unsupported_full128_rows(artifact, source_hash)
            kind = "FULL128_UNSUPPORTED_ARTIFACT"
        if records:
            declarations.append(
                RoleExposureDeclaration(
                    source_artifact_sha256=source_hash,
                    kind=ExposureDeclarationKind.PRIOR_ASSIGNMENT,
                    revoked=False,
                    records=records,
                )
            )
        unresolved.extend(failures)
        source_bindings.append(
            {
                "kind": kind,
                "source_artifact_sha256": source_hash,
                "schema_version": schema,
                "resolved_record_count": len(records),
                "unresolved_record_count": len(failures),
            }
        )

    for report, kfold_payload in masked_afn_runs:
        source_hash = content_sha256(report)
        records, failures, kfold_sha256 = _project_masked_afn(
            report,
            kfold_payload,
            source_hash=source_hash,
            public_source_bundle_sha256=bridge["binding"][
                "public_source_bundle_sha256"
            ],
            by_public=by_public,
        )
        nonparticipating = tuple(
            row
            for row in failures
            if row["reason"] == "KFOLD_SAMPLE_EXPLICITLY_NOT_PARTICIPATING"
        )
        failures = [
            row
            for row in failures
            if row["reason"] != "KFOLD_SAMPLE_EXPLICITLY_NOT_PARTICIPATING"
        ]
        if records:
            declarations.append(
                RoleExposureDeclaration(
                    source_artifact_sha256=source_hash,
                    kind=ExposureDeclarationKind.PRIOR_EVALUATION,
                    revoked=False,
                    records=records,
                )
            )
        unresolved.extend(failures)
        source_bindings.append(
            {
                "kind": "MASKED_AFN_KFOLD_REPORT",
                "source_artifact_sha256": source_hash,
                "schema_version": report.get("schema_version"),
                "kfold_manifest_sha256": kfold_sha256,
                "resolved_record_count": len(records),
                "nonparticipating_record_count": len(nonparticipating),
                "unresolved_record_count": len(failures),
            }
        )

    source_hashes = [item["source_artifact_sha256"] for item in source_bindings]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("face exposure source artifacts must be unique")
    unresolved.sort(
        key=lambda row: (
            row["source_artifact_sha256"],
            row["route_sample_token"] or "",
            row["public_sample_token"] or "",
            row["reason"],
        )
    )
    ledger = merge_role_exposure_declarations(declarations) if declarations else None
    receipt = create_role_exposure_receipt(ledger) if ledger is not None else None
    status = (
        "COMPLETE_EXACT_PROJECTIONS"
        if not unresolved
        else "BLOCKED_UNRESOLVED_PROJECTIONS"
    )
    history = {
        "schema_version": HISTORY_SCHEMA,
        "source_token_bridge_sha256": bridge["binding_sha256"],
        "source_token_bridge_bundle_sha256": bridge["bundle_sha256"],
        "public_source_bundle_sha256": bridge["binding"]["public_source_bundle_sha256"],
        "source_artifacts": sorted(
            source_bindings,
            key=lambda row: (row["source_artifact_sha256"], row["kind"]),
        ),
        "ledger": None if ledger is None else ledger.to_dict(),
        "ledger_sha256": None if ledger is None else ledger.ledger_sha256,
        "receipt": None if receipt is None else receipt.to_dict(),
        "receipt_sha256": None if receipt is None else receipt.receipt_sha256,
        "unresolved_rows": unresolved,
        "status": status,
        "role_allocation_permitted": not unresolved,
        "clean_role_claims_permitted": False,
        "final_evaluation_permitted": False,
        "interpretation": _INTERPRETATION,
    }
    history = {**history, "history_sha256": content_sha256(history)}
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "history": history,
        "history_sha256": history["history_sha256"],
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_face_exposure_history_bundle(value: object) -> dict[str, Any]:
    """Validate source bindings, unresolved blockers, and ledger/receipt closure."""

    expected = {"schema_version", "history", "history_sha256", "bundle_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("face exposure history bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("face exposure history bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("face exposure history bundle digest differs")
    history = bundle["history"]
    expected_history = {
        "schema_version",
        "source_token_bridge_sha256",
        "source_token_bridge_bundle_sha256",
        "public_source_bundle_sha256",
        "source_artifacts",
        "ledger",
        "ledger_sha256",
        "receipt",
        "receipt_sha256",
        "unresolved_rows",
        "status",
        "role_allocation_permitted",
        "clean_role_claims_permitted",
        "final_evaluation_permitted",
        "interpretation",
        "history_sha256",
    }
    if not isinstance(history, dict) or set(history) != expected_history:
        raise ValueError("face exposure history fields differ")
    if (
        history["schema_version"] != HISTORY_SCHEMA
        or history["clean_role_claims_permitted"] is not False
        or history["final_evaluation_permitted"] is not False
        or history["interpretation"] != _INTERPRETATION
    ):
        raise ValueError("face exposure history policy differs")
    for field in (
        "source_token_bridge_sha256",
        "source_token_bridge_bundle_sha256",
        "public_source_bundle_sha256",
        "history_sha256",
    ):
        _require_sha256(history[field], field)
    history_payload = {
        key: item for key, item in history.items() if key != "history_sha256"
    }
    if (
        history["history_sha256"] != content_sha256(history_payload)
        or bundle["history_sha256"] != history["history_sha256"]
    ):
        raise ValueError("face exposure history digest differs")
    sources = _validate_source_artifacts(history["source_artifacts"])
    unresolved = _validate_unresolved(history["unresolved_rows"])
    expected_status = (
        "COMPLETE_EXACT_PROJECTIONS"
        if not unresolved
        else "BLOCKED_UNRESOLVED_PROJECTIONS"
    )
    if (
        history["status"] != expected_status
        or history["role_allocation_permitted"] is not (not unresolved)
        or sum(item["unresolved_record_count"] for item in sources) != len(unresolved)
    ):
        raise ValueError("face exposure unresolved status differs")
    if history["ledger"] is None:
        if any(
            history[field] is not None
            for field in ("ledger_sha256", "receipt", "receipt_sha256")
        ):
            raise ValueError("face exposure null ledger bindings differ")
        if any(item["resolved_record_count"] for item in sources):
            raise ValueError("face exposure source counts require a ledger")
    else:
        if not isinstance(history["receipt"], dict):
            raise TypeError("face exposure receipt must be an object")
        ledger = RoleExposureLedger.from_dict(history["ledger"])
        receipt = RoleExposureReceipt.from_dict(history["receipt"])
        verify_role_exposure_receipt(ledger, receipt)
        if (
            history["ledger_sha256"] != ledger.ledger_sha256
            or history["receipt_sha256"] != receipt.receipt_sha256
            or set(receipt.source_artifact_sha256s)
            != {
                item["source_artifact_sha256"]
                for item in sources
                if item["resolved_record_count"]
            }
        ):
            raise ValueError("face exposure ledger or receipt binding differs")
        if sum(item["resolved_record_count"] for item in sources) != sum(
            len(item.records) for item in ledger.declarations
        ):
            raise ValueError("face exposure resolved source counts differ")
    return bundle


def _project_full128_variant(
    artifact: Mapping[str, Any],
    *,
    source_hash: str,
    by_route: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[RoleExposureDeclarationRecord, ...], list[dict[str, Any]]]:
    expected = {
        "schema_version",
        "variant_id",
        "method",
        "initialization",
        "bindings",
        "fit_population",
        "training",
        "artifacts",
        "variant_run_sha256",
    }
    if set(artifact) != expected:
        raise ValueError("Full128 variant run fields differ")
    payload = {
        key: item for key, item in artifact.items() if key != "variant_run_sha256"
    }
    if artifact["variant_run_sha256"] != content_sha256(payload):
        raise ValueError("Full128 variant run digest differs")
    fit = artifact["fit_population"]
    if not isinstance(fit, Mapping) or set(fit) != {
        "partition",
        "sample_count",
        "identity_count",
        "samples",
        "fit_population_sha256",
    }:
        raise ValueError("Full128 fit population fields differ")
    fit_payload = {
        key: item for key, item in fit.items() if key != "fit_population_sha256"
    }
    rows = fit.get("samples")
    if (
        fit["partition"] != "FIT"
        or fit["fit_population_sha256"] != content_sha256(fit_payload)
        or not isinstance(rows, list)
        or fit["sample_count"] != len(rows)
    ):
        raise ValueError("Full128 fit population digest or count differs")
    resolved: dict[str, RoleExposureDeclarationRecord] = {}
    unresolved: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "identity_id",
            "dataset_name",
            "view",
            "crop_record_sha256",
        }:
            raise ValueError("Full128 fit population sample fields differ")
        if row["dataset_name"] not in {"dogfacenet224", "mpdd"}:
            continue
        route_token = row["sample_id"]
        bridge = by_route.get(route_token)
        if bridge is None:
            unresolved.append(
                _unresolved(
                    source_hash,
                    "FULL128_VARIANT_RUN",
                    "FIT_SAMPLE_ABSENT_FROM_FACE_TOKEN_BRIDGE",
                    route_sample_token=route_token,
                )
            )
            continue
        if (
            row["dataset_name"] != bridge["dataset_name"]
            or row["identity_id"] != bridge["registered_identity_id"]
        ):
            raise ValueError("Full128 fit population identity binding differs")
        resolved[bridge["public_sample_token"]] = _declaration_record(
            bridge, ExposureStage.MODEL_TRAINING_USED
        )
    return tuple(
        sorted(resolved.values(), key=lambda row: row.sample_token)
    ), unresolved


def _unsupported_full128_rows(
    artifact: Mapping[str, Any], source_hash: str
) -> list[dict[str, Any]]:
    candidates: list[str | None] = []
    protocol = artifact.get("protocol")
    if isinstance(protocol, Mapping) and isinstance(
        protocol.get("sample_assignments"), list
    ):
        candidates.extend(
            row.get("sample_token") if isinstance(row, Mapping) else None
            for row in protocol["sample_assignments"]
        )
    if not candidates:
        candidates.append(None)
    return [
        _unresolved(
            source_hash,
            "FULL128_UNSUPPORTED_ARTIFACT",
            "ARTIFACT_DOES_NOT_PROVE_EXACT_EXECUTED_SAMPLE_PARTICIPATION",
            route_sample_token=token,
        )
        for token in candidates
    ]


def _project_masked_afn(
    report: Mapping[str, Any],
    kfold_payload: Mapping[str, Any],
    *,
    source_hash: str,
    public_source_bundle_sha256: str,
    by_public: Mapping[str, Mapping[str, Any]],
) -> tuple[
    tuple[RoleExposureDeclarationRecord, ...],
    list[dict[str, Any]],
    str,
]:
    expected = {
        "schema_version",
        "kfold_manifest_sha256",
        "candidate_manifest_sha256s",
        "candidate_source_token_binding",
        "config",
        "checkpoints",
        "folds",
        "retrieval_eligibility",
        "out_of_fold_test",
        "interpretation",
        "report_sha256",
    }
    if set(report) != expected or report["schema_version"] != (
        "cvi.masked_afn_kfold_report.v1"
    ):
        raise ValueError("Masked-AFN report schema or fields differ")
    body = {key: item for key, item in report.items() if key != "report_sha256"}
    if report["report_sha256"] != content_sha256(body):
        raise ValueError("Masked-AFN report digest differs")
    kfold_manifest_payload = kfold_payload
    if kfold_payload.get("schema_version") == (
        "cvi.dataset_stratified_identity_kfold_manifest_bundle.v1"
    ):
        if set(kfold_payload) != {"schema_version", "manifest_sha256", "manifest"}:
            raise ValueError("Masked-AFN K-fold bundle fields differ")
        kfold_manifest_payload = kfold_payload["manifest"]
        if (
            not isinstance(kfold_manifest_payload, Mapping)
            or kfold_payload["manifest_sha256"]
            != content_sha256(kfold_manifest_payload)
        ):
            raise ValueError("Masked-AFN K-fold bundle digest differs")
    kfold = DatasetStratifiedIdentityKFoldManifest.from_dict(kfold_manifest_payload)
    if report["kfold_manifest_sha256"] != kfold.manifest_sha256:
        raise ValueError("Masked-AFN report and K-fold manifest differ")
    folds = report["folds"]
    if not isinstance(folds, list) or sorted(
        row.get("fold_index") for row in folds if isinstance(row, Mapping)
    ) != list(range(kfold.policy.fold_count)):
        raise ValueError("Masked-AFN report fold coverage differs")
    sample_by_token = {
        row.sample_token: row
        for row in kfold.sample_assignments
        if row.source_variant == "original" and row.home_fold is not None
    }
    identity_by_token = {row.identity_token: row for row in kfold.identity_assignments}
    resolved: list[RoleExposureDeclarationRecord] = []
    unresolved: list[dict[str, Any]] = []
    binding = report["candidate_source_token_binding"]
    if (
        not isinstance(binding, Mapping)
        or binding.get("source_bundle_sha256") != public_source_bundle_sha256
    ):
        unresolved.extend(
            _unresolved(
                source_hash,
                "MASKED_AFN_KFOLD_REPORT",
                "REPORT_LACKS_MATCHING_PUBLIC_SOURCE_TOKEN_BINDING",
                route_sample_token=bridge["route_sample_token"],
                public_sample_token=public_token,
            )
            for public_token, bridge in sorted(by_public.items())
        )
        return (), unresolved, kfold.manifest_sha256
    for public_token, bridge in sorted(by_public.items()):
        sample = sample_by_token.get(public_token)
        if sample is None:
            unresolved.append(
                _unresolved(
                    source_hash,
                    "MASKED_AFN_KFOLD_REPORT",
                    "FACE_SAMPLE_ABSENT_FROM_EXECUTED_KFOLD_POPULATION",
                    route_sample_token=bridge["route_sample_token"],
                    public_sample_token=public_token,
                )
            )
            continue
        identity = identity_by_token[sample.identity_token]
        if (
            identity.registered_dog_id != bridge["registered_identity_id"]
            or identity.identity_token != bridge["public_identity_token"]
        ):
            raise ValueError("Masked-AFN K-fold face identity binding differs")
        if sample.held_out_role in {
            HeldOutSampleRole.GALLERY,
            HeldOutSampleRole.QUERY,
        }:
            stage = ExposureStage.MODEL_SELECTION_SCORED
        elif sample.training_eligible:
            stage = ExposureStage.MODEL_TRAINING_USED
        else:
            unresolved.append(
                _unresolved(
                    source_hash,
                    "MASKED_AFN_KFOLD_REPORT",
                    "KFOLD_SAMPLE_EXPLICITLY_NOT_PARTICIPATING",
                    route_sample_token=bridge["route_sample_token"],
                    public_sample_token=public_token,
                )
            )
            continue
        resolved.append(_declaration_record(bridge, stage))
    return (
        tuple(sorted(resolved, key=lambda row: row.sample_token)),
        unresolved,
        kfold.manifest_sha256,
    )


def _declaration_record(
    bridge: Mapping[str, Any], stage: ExposureStage
) -> RoleExposureDeclarationRecord:
    return RoleExposureDeclarationRecord(
        sample_token=bridge["public_sample_token"],
        identity_token=bridge["public_identity_token"],
        public_subject_token=bridge["public_subject_token"],
        stage=stage,
    )


def _unresolved(
    source_hash: str,
    artifact_kind: str,
    reason: str,
    *,
    route_sample_token: str | None = None,
    public_sample_token: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": UNRESOLVED_SCHEMA,
        "source_artifact_sha256": source_hash,
        "artifact_kind": artifact_kind,
        "route_sample_token": route_sample_token,
        "public_sample_token": public_sample_token,
        "reason": reason,
        "blocks_role_allocation": True,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _validate_source_artifacts(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("face exposure source artifacts must not be empty")
    rows: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) not in (
            {
                "kind",
                "source_artifact_sha256",
                "schema_version",
                "resolved_record_count",
                "unresolved_record_count",
            },
            {
                "kind",
                "source_artifact_sha256",
                "schema_version",
                "kfold_manifest_sha256",
                "resolved_record_count",
                "nonparticipating_record_count",
                "unresolved_record_count",
            },
        ):
            raise ValueError("face exposure source artifact fields differ")
        row = dict(item)
        _require_sha256(row["source_artifact_sha256"], "source artifact SHA-256")
        if "kfold_manifest_sha256" in row:
            _require_sha256(row["kfold_manifest_sha256"], "K-fold manifest SHA-256")
        for field in (
            "resolved_record_count",
            "nonparticipating_record_count",
            "unresolved_record_count",
        ):
            if field not in row:
                continue
            if (
                isinstance(row[field], bool)
                or not isinstance(row[field], int)
                or row[field] < 0
            ):
                raise ValueError("face exposure source artifact count differs")
        if row["source_artifact_sha256"] in hashes:
            raise ValueError("face exposure source artifact hashes repeat")
        hashes.add(row["source_artifact_sha256"])
        rows.append(row)
    if rows != sorted(
        rows, key=lambda row: (row["source_artifact_sha256"], row["kind"])
    ):
        raise ValueError("face exposure source artifacts are not sorted")
    return tuple(rows)


def _validate_unresolved(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise TypeError("face exposure unresolved rows must be an array")
    expected = {
        "schema_version",
        "source_artifact_sha256",
        "artifact_kind",
        "route_sample_token",
        "public_sample_token",
        "reason",
        "blocks_role_allocation",
        "record_sha256",
    }
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("face exposure unresolved row fields differ")
        row = dict(item)
        if (
            row["schema_version"] != UNRESOLVED_SCHEMA
            or row["blocks_role_allocation"] is not True
            or not isinstance(row["artifact_kind"], str)
            or not isinstance(row["reason"], str)
            or not row["reason"]
        ):
            raise ValueError("face exposure unresolved row policy differs")
        _require_sha256(row["source_artifact_sha256"], "source artifact SHA-256")
        _require_sha256(row["record_sha256"], "unresolved record SHA-256")
        for field in ("route_sample_token", "public_sample_token"):
            if row[field] is not None:
                _require_sha256(row[field], field)
        payload = {key: item for key, item in row.items() if key != "record_sha256"}
        if row["record_sha256"] != content_sha256(payload):
            raise ValueError("face exposure unresolved row digest differs")
        rows.append(row)
    expected_order = sorted(
        rows,
        key=lambda row: (
            row["source_artifact_sha256"],
            row["route_sample_token"] or "",
            row["public_sample_token"] or "",
            row["reason"],
        ),
    )
    if rows != expected_order:
        raise ValueError("face exposure unresolved rows are not sorted")
    return tuple(rows)


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "build_face_exposure_history",
    "validate_face_exposure_history_bundle",
]
