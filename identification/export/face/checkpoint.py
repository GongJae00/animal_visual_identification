"""FaceID checkpoint bindings and evaluation partition validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shared.contracts.source_provenance import build_source_provenance
from shared.foundation.provenance import content_sha256

CHECKPOINT_SCHEMA = "identification.face.faceid_checkpoint.v2"
_LEGACY_SOURCE_SHA256 = {
    "regional_v4": (
        "fa27c749ab879ad55d73a4a01f4ec84694984ff630187bfb5927c743de1cfa76",
        "630fb61fa624fbd3b5765cd178451e7c7747bd51b41e5b75584724c01bc6d174",
    ),
    "cls_residual_v5": (
        "4801633230047cc837247fd9dabe86304558c72f4480dc3079559293b6ce7611",
        "630fb61fa624fbd3b5765cd178451e7c7747bd51b41e5b75584724c01bc6d174",
    ),
    "aligned_cls_residual_v5": (
        "4801633230047cc837247fd9dabe86304558c72f4480dc3079559293b6ce7611",
        "630fb61fa624fbd3b5765cd178451e7c7747bd51b41e5b75584724c01bc6d174",
    ),
}
DINO_CONTRACT_KEYS = {
    "config_sha256",
    "model_sha256",
    "preprocessor_receipt_sha256",
    "preprocessor_sha256",
    "preprocessor_source_contract_sha256",
    "weight_receipt_sha256",
    "weight_source_contract_sha256",
}
CHECKPOINT_BINDING_KEYS = {
    "schema_version",
    "training_roi_manifest_sha256",
    "training_identity_ids",
    "dino_local_artifact_contract",
    "dino_local_artifact_contract_sha256",
    "weight_intake_bundle_sha256",
    "preprocessor_intake_bundle_sha256",
    "faceid_contract",
    "faceid_contract_sha256",
}
CHECKPOINT_KEYS = {
    *CHECKPOINT_BINDING_KEYS,
    "epoch",
    "encoder_state_dict",
    "quality_head_state_dict",
    "objective_state_dict",
    "optimizer_state_dict",
    "identity_to_index",
    "training_split_sha256",
    "checkpoint_bindings_sha256",
    "MRR",
    "Rank-1",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def normalize_dino_local_artifact_contract(
    fields: Mapping[str, object],
) -> dict[str, str]:
    if set(fields) != DINO_CONTRACT_KEYS:
        raise ValueError("DINO local artifact contract keys differ")
    return {
        name: require_sha256(fields[name], name) for name in sorted(DINO_CONTRACT_KEYS)
    }


def build_faceid_contract(
    architecture_source_sha256: str,
    input_source_sha256: str,
    *,
    architecture: str = "regional_v4",
) -> dict[str, Any]:
    if architecture in {"cls_residual_v5", "aligned_cls_residual_v5"}:
        return {
            "schema_version": "identification.face.architecture_input_contract.v2",
            "architecture": architecture,
            "architecture_source_sha256": require_sha256(
                architecture_source_sha256, "architecture_source_sha256"
            ),
            "input_source_sha256": require_sha256(
                input_source_sha256, "input_source_sha256"
            ),
            "backbone": {
                "model_id": "facebook/dinov2-small",
                "hidden_dimension": 384,
                "frozen": True,
            },
            "inputs": {
                "rgb_shape": ["batch", 3, 224, 224],
                "rgb_range": [0.0, 1.0],
                "alignment": (
                    "EYE_EYE_NOSE_AFFINE_OR_DETERMINISTIC_RAW_FALLBACK"
                    if architecture == "aligned_cls_residual_v5"
                    else "NONE_RAW_FACE_ROI"
                ),
                "minimum_anchor_confidence": 0.1,
            },
            "encoder": {
                "type": "ZERO_INITIALIZED_BOUNDED_CLS_RESIDUAL",
                "output_embedding_dimension": 384,
                "residual_scale": 0.1,
                "output_semantics": "L2_NORMALIZED",
                "quality_output_semantics": "SIGMOID_SCALAR",
            },
        }
    if architecture != "regional_v4":
        raise ValueError("unsupported FaceID contract architecture")
    return {
        "schema_version": "identification.face.architecture_input_contract.v1",
        "architecture_source_sha256": require_sha256(
            architecture_source_sha256, "architecture_source_sha256"
        ),
        "input_source_sha256": require_sha256(
            input_source_sha256, "input_source_sha256"
        ),
        "backbone": {
            "model_id": "facebook/dinov2-small",
            "patch_size": 14,
            "hidden_dimension": 384,
            "hidden_state_indices": [8, 9, 10, 11],
            "patch_grid": [16, 16],
            "frozen": True,
        },
        "inputs": {
            "rgb_shape": ["batch", 3, 224, 224],
            "rgb_range": [0.0, 1.0],
            "landmarks_shape": ["batch", 17, 3],
            "landmark_semantics": "face-roi-normalized-x-y-confidence",
        },
        "encoder": {
            "regional_embedding_dimension": 256,
            "output_embedding_dimension": 640,
            "baseline_embedding_dimension": 384,
            "regional_scale": 0.25,
            "region_centers": [
                [0.50, 0.50, 0.40, 0.40],
                [0.35, 0.35, 0.15, 0.12],
                [0.65, 0.35, 0.15, 0.12],
                [0.50, 0.50, 0.18, 0.22],
                [0.50, 0.25, 0.22, 0.14],
            ],
            "output_semantics": "L2_NORMALIZED_BASELINE_REGIONAL_CONCATENATION",
            "quality_output_semantics": "SIGMOID_SCALAR",
        },
    }


def build_faceid_source_contract(
    repository: Path,
    *,
    architecture: str,
) -> dict[str, Any]:
    """Build the current path-independent Face source contract."""

    if architecture not in _LEGACY_SOURCE_SHA256:
        raise ValueError("unsupported FaceID contract architecture")
    implementation = repository / "identification" / "export" / "face" / (
        "residual_model.py" if architecture != "regional_v4" else "model.py"
    )
    source = build_source_provenance(
        (
            implementation,
            repository / "identification" / "training" / "face" / "dataset.py",
        ),
        logical_component=f"embedding.methods.face.{architecture}",
    )
    legacy_shape = build_faceid_contract("0" * 64, "0" * 64, architecture=architecture)
    legacy_shape.pop("architecture_source_sha256")
    legacy_shape.pop("input_source_sha256")
    legacy_shape["schema_version"] = "identification.face.architecture_input_contract.v3"
    legacy_shape["source_provenance"] = source
    legacy_shape["source_provenance_sha256"] = content_sha256(source)
    return legacy_shape


def expected_faceid_contract_for_checkpoint(
    checkpoint_contract: Mapping[str, Any],
    repository: Path,
    *,
    architecture: str,
) -> dict[str, Any]:
    """Resolve current v3 or the concrete pre-move F4/F5 contract."""

    schema = checkpoint_contract.get("schema_version")
    if schema == "identification.face.architecture_input_contract.v3":
        return build_faceid_source_contract(repository, architecture=architecture)
    architecture_sha256, input_sha256 = _LEGACY_SOURCE_SHA256[architecture]
    return build_faceid_contract(
        architecture_sha256,
        input_sha256,
        architecture=architecture,
    )


def build_checkpoint_bindings(
    *,
    dino_local_artifact_contract: Mapping[str, object],
    weight_intake_bundle_sha256: str,
    preprocessor_intake_bundle_sha256: str,
    faceid_contract: Mapping[str, Any],
    training_roi_manifest_sha256: str,
    training_identity_ids: Sequence[str],
) -> dict[str, Any]:
    dino_contract = normalize_dino_local_artifact_contract(dino_local_artifact_contract)
    identities = list(training_identity_ids)
    if (
        not identities
        or any(not isinstance(identity, str) or not identity for identity in identities)
        or identities != sorted(set(identities))
    ):
        raise ValueError("training identity list must be non-empty, sorted, and unique")
    faceid = dict(faceid_contract)
    bindings = {
        "schema_version": CHECKPOINT_SCHEMA,
        "dino_local_artifact_contract": dino_contract,
        "dino_local_artifact_contract_sha256": content_sha256(dino_contract),
        "weight_intake_bundle_sha256": require_sha256(
            weight_intake_bundle_sha256, "weight_intake_bundle_sha256"
        ),
        "preprocessor_intake_bundle_sha256": require_sha256(
            preprocessor_intake_bundle_sha256,
            "preprocessor_intake_bundle_sha256",
        ),
        "faceid_contract": faceid,
        "faceid_contract_sha256": content_sha256(faceid),
        "training_roi_manifest_sha256": require_sha256(
            training_roi_manifest_sha256, "training_roi_manifest_sha256"
        ),
        "training_identity_ids": identities,
    }
    bindings["checkpoint_bindings_sha256"] = content_sha256(bindings)
    return bindings


def validate_checkpoint_structure(
    checkpoint: Mapping[str, Any],
    *,
    expected_faceid_contract: Mapping[str, Any],
) -> tuple[str, ...]:
    if set(checkpoint) != CHECKPOINT_KEYS:
        raise ValueError("FaceID checkpoint keys differ")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("FaceID checkpoint schema differs")
    for name in (
        "training_split_sha256",
        "training_roi_manifest_sha256",
        "dino_local_artifact_contract_sha256",
        "weight_intake_bundle_sha256",
        "preprocessor_intake_bundle_sha256",
        "faceid_contract_sha256",
        "checkpoint_bindings_sha256",
    ):
        require_sha256(checkpoint[name], name)
    dino_contract = checkpoint["dino_local_artifact_contract"]
    if not isinstance(dino_contract, Mapping):
        raise ValueError("checkpoint DINO local artifact contract must be an object")
    normalized_dino = normalize_dino_local_artifact_contract(dino_contract)
    if normalized_dino != dino_contract or (
        content_sha256(normalized_dino)
        != checkpoint["dino_local_artifact_contract_sha256"]
    ):
        raise ValueError("checkpoint DINO local artifact contract hash differs")

    faceid_contract = checkpoint["faceid_contract"]
    if not isinstance(faceid_contract, Mapping) or (
        content_sha256(faceid_contract) != checkpoint["faceid_contract_sha256"]
    ):
        raise ValueError("checkpoint FaceID contract hash differs")
    if faceid_contract != expected_faceid_contract:
        raise ValueError("checkpoint FaceID architecture/input contract differs")

    identities = checkpoint["training_identity_ids"]
    if (
        not isinstance(identities, list)
        or not identities
        or any(not isinstance(identity, str) or not identity for identity in identities)
        or identities != sorted(set(identities))
    ):
        raise ValueError("checkpoint training identity list is not exact and canonical")
    expected_index = {identity: index for index, identity in enumerate(identities)}
    if checkpoint["identity_to_index"] != expected_index:
        raise ValueError(
            "checkpoint identity index differs from training identity list"
        )
    binding_payload = {name: checkpoint[name] for name in CHECKPOINT_BINDING_KEYS}
    if content_sha256(binding_payload) != checkpoint["checkpoint_bindings_sha256"]:
        raise ValueError("FaceID checkpoint aggregate binding hash differs")
    return tuple(identities)


def validate_checkpoint_runtime_bindings(
    checkpoint: Mapping[str, Any],
    *,
    observed_dino_local_artifact_contract: Mapping[str, object],
    observed_weight_intake_bundle_sha256: str,
    observed_preprocessor_intake_bundle_sha256: str,
) -> None:
    observed_dino = normalize_dino_local_artifact_contract(
        observed_dino_local_artifact_contract
    )
    if checkpoint["dino_local_artifact_contract"] != observed_dino or checkpoint[
        "dino_local_artifact_contract_sha256"
    ] != content_sha256(observed_dino):
        raise ValueError("checkpoint DINO contract differs from local artifact")
    if checkpoint["weight_intake_bundle_sha256"] != require_sha256(
        observed_weight_intake_bundle_sha256,
        "observed_weight_intake_bundle_sha256",
    ):
        raise ValueError("checkpoint weight intake bundle differs")
    if checkpoint["preprocessor_intake_bundle_sha256"] != require_sha256(
        observed_preprocessor_intake_bundle_sha256,
        "observed_preprocessor_intake_bundle_sha256",
    ):
        raise ValueError("checkpoint preprocessor intake bundle differs")


def validate_evaluation_partition(
    manifest: Mapping[str, Any],
    *,
    training_roi_manifest_sha256: str,
    training_identity_ids: Sequence[str],
    expected_split_role: str,
) -> dict[str, Any]:
    if not isinstance(expected_split_role, str) or not expected_split_role:
        raise ValueError("expected split role must be a non-empty string")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation ROI manifest has no records")
    roles = {
        record.get("split_role") if isinstance(record, Mapping) else None
        for record in records
    }
    if roles != {expected_split_role}:
        raise ValueError(
            f"evaluation ROI split role differs: expected {expected_split_role!r}, "
            f"observed {sorted(repr(role) for role in roles)}"
        )

    evaluation_manifest_sha256 = content_sha256(manifest)
    if evaluation_manifest_sha256 == require_sha256(
        training_roi_manifest_sha256, "training_roi_manifest_sha256"
    ):
        raise ValueError("evaluation ROI manifest is the training ROI manifest")
    evaluation_identities = sorted(
        {
            record["registered_identity_id"]
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("registered_identity_id"), str)
            and record["registered_identity_id"]
        }
    )
    if not evaluation_identities:
        raise ValueError("evaluation ROI manifest has no registered identities")
    overlap = sorted(set(training_identity_ids) & set(evaluation_identities))
    if overlap:
        raise ValueError("training/evaluation identity overlap: " + ", ".join(overlap))
    return {
        "status": "verified",
        "training_and_evaluation_manifests_distinct": True,
        "training_and_evaluation_identities_disjoint": True,
        "expected_split_role": expected_split_role,
        "observed_split_roles": [expected_split_role],
        "evaluation_identity_count": len(evaluation_identities),
        "evaluation_roi_manifest_sha256": evaluation_manifest_sha256,
    }
