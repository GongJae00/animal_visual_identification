from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from embedding.methods.face import checkpoint as faceid_checkpoint
from foundation import provenance

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("status", "dirty"), [("", False), (" M tracked\n?? new\n", True)]
)
def test_git_worktree_provenance_uses_exact_commands_and_dirty_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    dirty: bool,
) -> None:
    calls: list[tuple[tuple[str, ...], bool, Path]] = []
    outputs = iter(("abc123\n", status))

    def check_output(command: tuple[str, ...], *, text: bool, cwd: Path) -> str:
        calls.append((command, text, cwd))
        return next(outputs)

    monkeypatch.setattr(provenance.subprocess, "check_output", check_output)

    assert provenance.git_worktree_provenance(tmp_path) == {
        "code_commit": "abc123",
        "worktree_dirty": dirty,
        "worktree_status_basis": (
            "git status --porcelain=v1 --untracked-files=normal; includes staged, "
            "unstaged, and untracked path status, not untracked file contents"
        ),
    }
    assert calls == [
        (("git", "rev-parse", "HEAD"), True, tmp_path),
        (
            ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
            True,
            tmp_path,
        ),
    ]


def test_git_worktree_provenance_propagates_git_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    error = subprocess.CalledProcessError(128, ("git", "rev-parse", "HEAD"))

    def check_output(*args: object, **kwargs: object) -> str:
        raise error

    monkeypatch.setattr(provenance.subprocess, "check_output", check_output)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        provenance.git_worktree_provenance(tmp_path)

    assert raised.value is error


def _dino_contract() -> dict[str, str]:
    return {
        "config_sha256": "0" * 64,
        "model_sha256": "a" * 64,
        "preprocessor_receipt_sha256": "b" * 64,
        "preprocessor_sha256": "c" * 64,
        "preprocessor_source_contract_sha256": "d" * 64,
        "weight_receipt_sha256": "e" * 64,
        "weight_source_contract_sha256": "f" * 64,
    }


def _faceid_contract() -> dict:
    return faceid_checkpoint.build_faceid_contract("1" * 64, "2" * 64)


def _checkpoint() -> dict:
    identities = ["training-dog-a", "training-dog-b"]
    bindings = faceid_checkpoint.build_checkpoint_bindings(
        dino_local_artifact_contract=_dino_contract(),
        weight_intake_bundle_sha256="3" * 64,
        preprocessor_intake_bundle_sha256="4" * 64,
        faceid_contract=_faceid_contract(),
        training_roi_manifest_sha256="5" * 64,
        training_identity_ids=identities,
    )
    return {
        **bindings,
        "epoch": 3,
        "encoder_state_dict": {},
        "quality_head_state_dict": {},
        "objective_state_dict": {},
        "optimizer_state_dict": {},
        "identity_to_index": {
            identity: index for index, identity in enumerate(identities)
        },
        "training_split_sha256": "6" * 64,
        "MRR": 0.5,
        "Rank-1": 0.4,
    }


def _manifest(*, role: str = "test", identity: str = "evaluation-dog") -> dict:
    return {
        "schema_version": "cvi.canid_roi_manifest.v2",
        "source_sample_ids": ["sample-a"],
        "records": [
            {
                "sample_id": "sample-a",
                "split_role": role,
                "registered_identity_id": identity,
            }
        ],
    }


def test_checkpoint_v2_binds_exact_contracts_and_training_identities() -> None:
    checkpoint = _checkpoint()

    identities = faceid_checkpoint.validate_checkpoint_structure(
        checkpoint, expected_faceid_contract=_faceid_contract()
    )
    faceid_checkpoint.validate_checkpoint_runtime_bindings(
        checkpoint,
        observed_dino_local_artifact_contract=_dino_contract(),
        observed_weight_intake_bundle_sha256="3" * 64,
        observed_preprocessor_intake_bundle_sha256="4" * 64,
    )

    assert checkpoint["schema_version"] == faceid_checkpoint.CHECKPOINT_SCHEMA
    assert identities == ("training-dog-a", "training-dog-b")
    assert checkpoint[
        "dino_local_artifact_contract_sha256"
    ] == faceid_checkpoint.content_sha256(_dino_contract())
    assert checkpoint["faceid_contract_sha256"] == faceid_checkpoint.content_sha256(
        _faceid_contract()
    )
    assert checkpoint["checkpoint_bindings_sha256"] == faceid_checkpoint.content_sha256(
        {name: checkpoint[name] for name in faceid_checkpoint.CHECKPOINT_BINDING_KEYS}
    )


def test_checkpoint_rejects_noncanonical_or_inconsistent_identity_list() -> None:
    checkpoint = _checkpoint()
    checkpoint["training_identity_ids"] = ["training-dog-b", "training-dog-a"]

    with pytest.raises(ValueError, match="identity list"):
        faceid_checkpoint.validate_checkpoint_structure(
            checkpoint, expected_faceid_contract=_faceid_contract()
        )

    checkpoint = _checkpoint()
    checkpoint["identity_to_index"] = {
        "training-dog-a": 1,
        "training-dog-b": 0,
    }
    with pytest.raises(ValueError, match="identity index"):
        faceid_checkpoint.validate_checkpoint_structure(
            checkpoint, expected_faceid_contract=_faceid_contract()
        )


def test_checkpoint_rejects_binding_payload_tampering() -> None:
    checkpoint = _checkpoint()
    checkpoint["training_roi_manifest_sha256"] = "7" * 64

    with pytest.raises(ValueError, match="aggregate binding hash"):
        faceid_checkpoint.validate_checkpoint_structure(
            checkpoint, expected_faceid_contract=_faceid_contract()
        )


def test_checkpoint_rejects_artifact_bundle_and_architecture_drift() -> None:
    checkpoint = _checkpoint()
    changed_dino = _dino_contract()
    changed_dino["model_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="local artifact"):
        faceid_checkpoint.validate_checkpoint_runtime_bindings(
            checkpoint,
            observed_dino_local_artifact_contract=changed_dino,
            observed_weight_intake_bundle_sha256="3" * 64,
            observed_preprocessor_intake_bundle_sha256="4" * 64,
        )

    with pytest.raises(ValueError, match="weight intake bundle"):
        faceid_checkpoint.validate_checkpoint_runtime_bindings(
            checkpoint,
            observed_dino_local_artifact_contract=_dino_contract(),
            observed_weight_intake_bundle_sha256="8" * 64,
            observed_preprocessor_intake_bundle_sha256="4" * 64,
        )

    with pytest.raises(ValueError, match="preprocessor intake bundle"):
        faceid_checkpoint.validate_checkpoint_runtime_bindings(
            checkpoint,
            observed_dino_local_artifact_contract=_dino_contract(),
            observed_weight_intake_bundle_sha256="3" * 64,
            observed_preprocessor_intake_bundle_sha256="8" * 64,
        )

    changed_checkpoint = deepcopy(checkpoint)
    changed_checkpoint["faceid_contract"]["encoder"]["embedding_dimension"] = 512
    changed_checkpoint["faceid_contract_sha256"] = faceid_checkpoint.content_sha256(
        changed_checkpoint["faceid_contract"]
    )
    with pytest.raises(ValueError, match="architecture/input"):
        faceid_checkpoint.validate_checkpoint_structure(
            changed_checkpoint, expected_faceid_contract=_faceid_contract()
        )


def test_partition_verification_proves_role_manifest_and_identity_separation() -> None:
    manifest = _manifest()
    audit = faceid_checkpoint.validate_evaluation_partition(
        manifest,
        training_roi_manifest_sha256="5" * 64,
        training_identity_ids=("training-dog-a", "training-dog-b"),
        expected_split_role="test",
    )

    assert audit["status"] == "verified"
    assert audit["training_and_evaluation_manifests_distinct"] is True
    assert audit["training_and_evaluation_identities_disjoint"] is True
    assert audit["observed_split_roles"] == ["test"]


def test_partition_rejects_same_manifest_identity_overlap_and_role_mismatch() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="training ROI manifest"):
        faceid_checkpoint.validate_evaluation_partition(
            manifest,
            training_roi_manifest_sha256=faceid_checkpoint.content_sha256(manifest),
            training_identity_ids=("training-dog",),
            expected_split_role="test",
        )

    with pytest.raises(ValueError, match="identity overlap"):
        faceid_checkpoint.validate_evaluation_partition(
            _manifest(identity="training-dog"),
            training_roi_manifest_sha256="5" * 64,
            training_identity_ids=("training-dog",),
            expected_split_role="test",
        )

    with pytest.raises(ValueError, match="split role differs"):
        faceid_checkpoint.validate_evaluation_partition(
            _manifest(role="val"),
            training_roi_manifest_sha256="5" * 64,
            training_identity_ids=("training-dog",),
            expected_split_role="test",
        )


@pytest.mark.parametrize(
    "script, expected_help",
    [
        ("legacy/version/face/workflows/train_roi_face_reid.py", "{cpu,cuda}"),
        ("legacy/version/face/workflows/evaluate_trained_face_reid.py", "--expected-split-role"),
    ],
)
def test_help_does_not_import_model_runtime_or_create_outputs(
    tmp_path: Path,
    script: str,
    expected_help: str,
) -> None:
    guarded = """
import builtins
import runpy
import sys

blocked = {"cvi", "numpy", "torch", "transformers"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"blocked import reached: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
script = sys.argv[1]
sys.argv = [script, "--help"]
runpy.run_path(script, run_name="__main__")
"""
    completed = subprocess.run(
        [sys.executable, "-c", guarded, script],
        cwd=ROOT,
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_help in completed.stdout
    assert "blocked import reached" not in completed.stderr
    assert list(tmp_path.iterdir()) == []
