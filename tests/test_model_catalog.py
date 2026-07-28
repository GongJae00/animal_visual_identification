from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cvi.model_catalog import (
    MODEL_CATALOG,
    MODEL_ROLE_ALIASES,
    ModelAdmission,
    ModelArtifact,
    get_model_artifact,
    verify_model_artifact,
)


def test_catalog_ids_paths_and_aliases_are_consistent() -> None:
    ids = {artifact.artifact_id for artifact in MODEL_CATALOG}
    assert len(ids) == len(MODEL_CATALOG)
    assert set(MODEL_ROLE_ALIASES.values()) <= ids
    for artifact in MODEL_CATALOG:
        assert Path(artifact.relative_path).parts[0] == artifact.artifact_id


def test_roles_resolve_without_filesystem_symlinks() -> None:
    detector = get_model_artifact("dog-detector")
    assert detector.artifact_id == "yolo11n-pose-ap10k-dog-v2-20260728"
    assert detector.admission is ModelAdmission.RESEARCH_ONLY
    assert get_model_artifact("dog-pose") is detector
    assert get_model_artifact(detector.artifact_id) is detector


def test_agpl_artifact_is_not_a_deployment_candidate() -> None:
    artifact = get_model_artifact("dog-detector-coco")
    assert artifact.license_id == "AGPL-3.0-only"
    assert artifact.admission is ModelAdmission.RESEARCH_ONLY


def test_unknown_role_fails_closed() -> None:
    with pytest.raises(KeyError, match="unknown model artifact or role"):
        get_model_artifact("missing-role")


def test_verify_model_artifact_checks_exact_bytes(tmp_path: Path) -> None:
    payload = b"fixture model bytes"
    artifact = ModelArtifact(
        artifact_id="fixture-model-v1",
        relative_path="fixture-model-v1/model.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        source_model_id="fixture/model",
        source_revision="v1",
        license_id="TEST-ONLY",
        admission=ModelAdmission.RESEARCH_ONLY,
    )
    model_dir = tmp_path / artifact.artifact_id
    model_dir.mkdir()
    model_path = model_dir / "model.bin"
    model_path.write_bytes(payload)
    assert verify_model_artifact(artifact, tmp_path) == model_path

    model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_model_artifact(artifact, tmp_path)
