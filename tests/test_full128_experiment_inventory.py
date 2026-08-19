from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from data.source_lock import get_record
from foundation.provenance import content_sha256
from identity.full.full_split_census import (
    FullStatus,
    IdentityEvidenceKind,
    RegionStatus,
    TerminalRole,
    UnifiedFullObservation,
    ViewScope,
    allocate_unified_full_split,
    unified_full_split_bundle,
)
from identity.registry.generated_identity_registry import (
    GENERATED_DOG_NAMESPACE,
    compute_generated_identity_id,
    compute_source_cluster_token,
)
from identity.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_registered_dog_id,
)
from embedding.methods.full_segment.preparation import inventory
from embedding.methods.full_segment.preparation.data import load_full128_assembly
from embedding.methods.full_segment.preparation.inventory import (
    build_full128_experiment_inventory,
    validate_full128_experiment_inventory_bundle,
)
from embedding.methods.full_segment.preparation.materialization import ASSEMBLY_SCHEMA
from parsing.full_segment.full_segment_contracts import FullSegmentObservation
from legacy.version.full128.workflows.build_full128_experiment_inventory import (
    REQUEST_SCHEMA,
)
from legacy.version.full128.workflows.build_full128_experiment_inventory import (
    main as inventory_main,
)
from workflows.materialize_full_segment import (
    REQUEST_SCHEMA as MATERIALIZATION_REQUEST_SCHEMA,
)
from workflows.materialize_full_segment import run as materialize_full_segment


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _source_bytes(offset: int) -> bytes:
    values = (np.arange(8 * 12 * 3, dtype=np.uint16).reshape(8, 12, 3) + offset).astype(
        np.uint8
    )
    output = io.BytesIO()
    Image.fromarray(values, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _identity(kind: IdentityEvidenceKind, index: int) -> tuple[str | None, str | None]:
    if kind is IdentityEvidenceKind.REGISTERED:
        return (
            str(REGISTERED_DOG_NAMESPACE),
            compute_registered_dog_id(f"full128-fixture:{index}"),
        )
    if kind is IdentityEvidenceKind.GENERATED:
        cluster = compute_source_cluster_token(f"full128-cluster:{index}")
        return (
            str(GENERATED_DOG_NAMESPACE),
            compute_generated_identity_id("full128-fixture-generator:v1", cluster),
        )
    return None, None


def _build_case(
    artifact_root: Path,
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations: list[UnifiedFullObservation] = []
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        sample_token = _sha(f"sample:{index}")
        source_path = artifact_root / f"source-{index}.png"
        source_path.write_bytes(_source_bytes(index))
        output_dir = artifact_root / f"materialized-{index}"
        cache_bundle = materialize_full_segment(
            {
                "schema_version": MATERIALIZATION_REQUEST_SCHEMA,
                "source_id": sample_token,
                "source_image_path": str(source_path),
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "source_view_scope": "FACE_NATIVE",
                "route": "NATIVE_FACE",
                "frozen_parsing_path": None,
                "association": None,
                "face_observability": "NATIVE",
                "nose_observability": "NOT_DETECTED",
                "target_size": 32,
                "context_fraction": 0.0,
                "background_rgb": [127, 127, 127],
            },
            output_dir=output_dir,
        )
        full_observation = FullSegmentObservation.from_dict(
            cache_bundle["cache"]["records"][0]["observation"]
        )
        dataset = get_record(spec.get("dataset", "dogfacenet224"))
        kind = spec.get("kind", IdentityEvidenceKind.REGISTERED)
        namespace, identity_token = _identity(kind, spec.get("identity", index))
        gradient_eligible = dataset.admission.value == "ADMIT_TRAIN"
        validation_only = dataset.admission.value == "ADMIT_VALIDATION_ONLY"
        gradient_eligible = spec.get("gradient_eligible", gradient_eligible)
        observation = UnifiedFullObservation(
            dataset_name=dataset.canonical_name,
            official_split=spec.get("official_split", "official-train"),
            identity_evidence_kind=kind,
            identity_namespace_uuid=namespace,
            identity_token=identity_token,
            sample_token=sample_token,
            source_group=spec.get("source_group", f"source-group:{index}"),
            capture_group=spec.get("capture_group", f"capture-group:{index}"),
            sequence_group=spec.get("sequence_group", f"sequence-group:{index}"),
            duplicate_component=spec.get(
                "duplicate_component", _sha(f"duplicate:{index}")
            ),
            gradient_eligible=gradient_eligible,
            validation_only=validation_only,
            full_status=FullStatus(full_observation.full_status.value),
            face_status=RegionStatus(full_observation.face_observability.value),
            nose_status=RegionStatus(full_observation.nose_observability.value),
            view_scope=ViewScope(full_observation.source_view_scope.value),
            source_observation_sha256=full_observation.observation_sha256,
            terminal_role=spec.get("terminal_role"),
        )
        observations.append(observation)
        rows.append(
            {
                "dataset_name": dataset.canonical_name,
                "dataset_version": dataset.version,
                "official_split": observation.official_split,
                "identity_evidence_kind": kind.value,
                "identity_namespace_uuid": namespace,
                "identity_token": identity_token,
                "sample_token": sample_token,
                "source_group": observation.source_group,
                "capture_group": observation.capture_group,
                "sequence_group": observation.sequence_group,
                "duplicate_component": observation.duplicate_component,
                "terminal_role": None,
                "original_source_sha256": full_observation.source_sha256,
                "effective_source_sha256": full_observation.source_sha256,
                "lineage_receipt_path": None,
                "full_segment_cache_path": str(output_dir / "full-segment-cache.json"),
                "full_rgb_path": str(output_dir / "full.png"),
                "full_mask_path": str(output_dir / "full-mask.png"),
            }
        )
    manifest = allocate_unified_full_split(
        allocation_name="full128-test-allocation", observations=observations
    )
    role_by_sample = {
        observation.sample_token: observation.terminal_role.value
        for observation in manifest.observations
    }
    for row in rows:
        row["terminal_role"] = role_by_sample[row["sample_token"]]
    return unified_full_split_bundle(manifest), rows


def test_inventory_order_is_deterministic_and_verifies_native_artifacts(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}, {"identity": 1}])

    first = build_full128_experiment_inventory(
        unified_full_split=split,
        request_rows=rows,
        artifact_root=artifact_root,
    )
    second = build_full128_experiment_inventory(
        unified_full_split=split,
        request_rows=list(reversed(rows)),
        artifact_root=artifact_root,
        validation_workers=8,
    )

    assert first == second
    assert first["content_kind"] == "METADATA_ONLY"
    assert [
        record["sample_token"] for record in first["inventory"]["records"]
    ] == sorted(row["sample_token"] for row in rows)
    assert all(
        record["crop_artifacts_present"] for record in first["inventory"]["records"]
    )
    assert validate_full128_experiment_inventory_bundle(first) == first
    assert (
        validate_full128_experiment_inventory_bundle(first, validation_workers=8)
        == first
    )
    assert "embeddings" not in first and "metrics" not in first


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_validation_workers_must_be_positive_integers(
    tmp_path: Path, workers: object
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}])
    error = TypeError if isinstance(workers, bool | float) else ValueError

    with pytest.raises(error, match="validation workers"):
        build_full128_experiment_inventory(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
            validation_workers=workers,  # type: ignore[arg-type]
        )


def test_parallel_failure_is_the_earliest_corrupt_input_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}, {"identity": 1}])
    first_mask = Path(rows[0]["full_mask_path"])
    first_mask.unlink()
    second_rgb = Path(rows[1]["full_rgb_path"])
    second_rgb.write_bytes(second_rgb.read_bytes() + b"tampered")
    original = inventory._deep_validate_artifact

    def delayed_first(**kwargs: Any) -> Any:
        if kwargs["row"]["sample_token"] == rows[0]["sample_token"]:
            time.sleep(0.05)
        return original(**kwargs)

    monkeypatch.setattr(inventory, "_deep_validate_artifact", delayed_first)
    with pytest.raises(FileNotFoundError, match=str(first_mask)):
        build_full128_experiment_inventory(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
            validation_workers=8,
        )


def test_parallel_validation_rejects_crop_path_replacement(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}, {"identity": 1}])
    bundle = build_full128_experiment_inventory(
        unified_full_split=split,
        request_rows=rows,
        artifact_root=artifact_root,
        validation_workers=8,
    )
    first_rgb = Path(rows[0]["full_rgb_path"])
    second_rgb = Path(rows[1]["full_rgb_path"])
    first_rgb.write_bytes(second_rgb.read_bytes())

    with pytest.raises(ValueError, match="full_rgb artifact"):
        validate_full128_experiment_inventory_bundle(
            bundle,
            validation_workers=8,
        )


def test_assembly_load_returns_validated_bundle_without_a_second_read(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}, {"identity": 1}])
    bundle = build_full128_experiment_inventory(
        unified_full_split=split,
        request_rows=rows,
        artifact_root=artifact_root,
    )
    payload = {
        "schema_version": ASSEMBLY_SCHEMA,
        "plan_sha256": _sha("assembly-plan"),
        "sample_count": len(rows),
        "allocation_name": "full128-test-allocation",
        "topology_report": {},
        "unified_full_split": split,
        "inventory_request": {"schema_version": REQUEST_SCHEMA, "rows": rows},
        "inventory_bundle": bundle,
    }
    assembly = {**payload, "assembly_sha256": content_sha256(payload)}
    assembly_path = tmp_path / "assembly.json"
    assembly_path.write_text(json.dumps(assembly), encoding="utf-8")

    loaded, loaded_bundle = load_full128_assembly(
        assembly_path,
        validation_workers=8,
    )

    assert loaded_bundle == bundle
    assert loaded.inventory_bundle_sha256 == bundle["bundle_sha256"]
    assert [sample.sample_id for sample in loaded.samples] == sorted(
        row["sample_token"] for row in rows
    )


def test_plain_mapping_cannot_forge_prevalidated_materialization(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}])
    rgb_path = Path(rows[0]["full_rgb_path"])
    rgb_path.write_bytes(rgb_path.read_bytes() + b"tampered")

    with pytest.raises(TypeError, match="typed assembly evidence"):
        inventory._build_full128_experiment_inventory_from_prevalidated(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
            prevalidated={"artifacts": []},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="full_rgb artifact"):
        build_full128_experiment_inventory(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
        )


def test_blocked_petface_is_rejected_before_artifact_access(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    sample_token = _sha("blocked-sample")
    dataset = get_record("petface-dog")
    observation = UnifiedFullObservation(
        dataset_name=dataset.canonical_name,
        official_split="intake-only",
        identity_evidence_kind=IdentityEvidenceKind.NONE,
        identity_namespace_uuid=None,
        identity_token=None,
        sample_token=sample_token,
        source_group="blocked-source",
        capture_group="blocked-capture",
        sequence_group="blocked-sequence",
        duplicate_component=_sha("blocked-duplicate"),
        gradient_eligible=False,
        validation_only=False,
        full_status=FullStatus.UNUSABLE,
        face_status=RegionStatus.NOT_RUN,
        nose_status=RegionStatus.NOT_RUN,
        view_scope=ViewScope.UNAVAILABLE,
        source_observation_sha256=_sha("blocked-observation"),
        terminal_role=TerminalRole.BLOCKED,
    )
    manifest = allocate_unified_full_split(
        allocation_name="blocked-petface", observations=(observation,)
    )
    missing = artifact_root / "never-read"
    row = {
        "dataset_name": dataset.canonical_name,
        "dataset_version": dataset.version,
        "official_split": observation.official_split,
        "identity_evidence_kind": "NONE",
        "identity_namespace_uuid": None,
        "identity_token": None,
        "sample_token": sample_token,
        "source_group": observation.source_group,
        "capture_group": observation.capture_group,
        "sequence_group": observation.sequence_group,
        "duplicate_component": observation.duplicate_component,
        "terminal_role": "BLOCKED",
        "original_source_sha256": _sha("unread-source"),
        "effective_source_sha256": _sha("unread-source"),
        "lineage_receipt_path": None,
        "full_segment_cache_path": str(missing / "full-segment-cache.json"),
        "full_rgb_path": str(missing / "full.png"),
        "full_mask_path": str(missing / "full-mask.png"),
    }

    with pytest.raises(ValueError, match="blocked by source_lock"):
        build_full128_experiment_inventory(
            unified_full_split=unified_full_split_bundle(manifest),
            request_rows=[row],
            artifact_root=artifact_root,
        )


def test_missing_and_tampered_crop_artifacts_fail_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}])
    mask_path = Path(rows[0]["full_mask_path"])
    mask_bytes = mask_path.read_bytes()
    mask_path.unlink()
    with pytest.raises(FileNotFoundError, match="Full mask does not exist"):
        build_full128_experiment_inventory(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
        )

    mask_path.write_bytes(mask_bytes)
    rgb_path = Path(rows[0]["full_rgb_path"])
    rgb_path.write_bytes(rgb_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="full_rgb artifact byte size differs"):
        build_full128_experiment_inventory(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
        )


def test_source_digest_and_row_level_train_eligibility_are_bound(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(
        artifact_root,
        [
            {
                "official_split": "official-test",
                "gradient_eligible": False,
                "terminal_role": TerminalRole.EVAL,
            }
        ],
    )

    bundle = build_full128_experiment_inventory(
        unified_full_split=split,
        request_rows=rows,
        artifact_root=artifact_root,
    )
    assert bundle["inventory"]["records"][0]["gradient_eligible"] is False

    rows[0]["effective_source_sha256"] = _sha("substituted-source")
    with pytest.raises(ValueError, match="effective source digest differs"):
        build_full128_experiment_inventory(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
        )


@pytest.mark.parametrize(
    ("kind", "wrong_namespace"),
    [
        (IdentityEvidenceKind.REGISTERED, str(GENERATED_DOG_NAMESPACE)),
        (IdentityEvidenceKind.GENERATED, str(REGISTERED_DOG_NAMESPACE)),
    ],
)
def test_identity_uuid_namespaces_are_enforced(
    tmp_path: Path,
    kind: IdentityEvidenceKind,
    wrong_namespace: str,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{"kind": kind}])
    rows[0]["identity_namespace_uuid"] = wrong_namespace

    with pytest.raises(ValueError, match="identity UUID namespace"):
        build_full128_experiment_inventory(
            unified_full_split=split,
            request_rows=rows,
            artifact_root=artifact_root,
        )


def test_allocation_preserves_roles_and_duplicate_disjointness(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    duplicate = _sha("shared-duplicate")
    split, rows = _build_case(
        artifact_root,
        [
            {"identity": 10, "duplicate_component": duplicate},
            {
                "kind": IdentityEvidenceKind.GENERATED,
                "identity": 20,
                "duplicate_component": duplicate,
            },
            {"dataset": "dogflw", "kind": IdentityEvidenceKind.NONE},
            {
                "dataset": "mpdd",
                "identity": 30,
                "official_split": "official-validation",
            },
            {"identity": 40, "terminal_role": TerminalRole.EVAL},
        ],
    )
    bundle = build_full128_experiment_inventory(
        unified_full_split=split,
        request_rows=list(reversed(rows)),
        artifact_root=artifact_root,
    )
    records = {
        record["sample_token"]: record for record in bundle["inventory"]["records"]
    }

    assert (
        records[_sha("sample:0")]["terminal_role"]
        == records[_sha("sample:1")]["terminal_role"]
    )
    assert records[_sha("sample:2")]["terminal_role"] == "AUXILIARY"
    assert records[_sha("sample:2")]["identity_token"] is None
    assert records[_sha("sample:2")]["gradient_eligible"] is False
    assert records[_sha("sample:2")]["validation_only"] is False
    assert records[_sha("sample:3")]["gradient_eligible"] is False
    assert records[_sha("sample:3")]["validation_only"] is True
    assert records[_sha("sample:3")]["terminal_role"] != "FIT"
    assert records[_sha("sample:4")]["terminal_role"] == "EVAL"
    assert all(
        not values
        for values in bundle["split_bundle"]["census"]["overlap_report"].values()
    )


def test_workflow_publishes_once_and_round_trips(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    split, rows = _build_case(artifact_root, [{}])
    request_path = tmp_path / "request.json"
    split_path = tmp_path / "split.json"
    output_path = tmp_path / "full128-inventory.json"
    request_path.write_text(
        json.dumps({"schema_version": REQUEST_SCHEMA, "rows": rows}),
        encoding="utf-8",
    )
    split_path.write_text(json.dumps(split), encoding="utf-8")
    arguments = [
        "--request",
        str(request_path),
        "--unified-full-split",
        str(split_path),
        "--artifact-root",
        str(artifact_root),
        "--output",
        str(output_path),
    ]

    assert inventory_main(arguments) == 0
    published = json.loads(output_path.read_text(encoding="utf-8"))
    assert validate_full128_experiment_inventory_bundle(published) == published
    assert published["inventory_sha256"] == content_sha256(published["inventory"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        inventory_main(arguments)
