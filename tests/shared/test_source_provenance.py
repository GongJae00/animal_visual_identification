from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from shared.contracts.source_provenance import (
    build_offline_tool_provenance,
    build_source_provenance,
)
from identification.export.face.checkpoint import (
    build_faceid_source_contract,
    expected_faceid_contract_for_checkpoint,
)

from tests.repo_root import REPO_ROOT as ROOT

def test_v2_source_closure_is_recursive_logical_and_deterministic() -> None:
    tool = ROOT / "evaluation" / "splits" / "registry_cli.py"
    first = build_offline_tool_provenance(tool)
    second = build_offline_tool_provenance(tool)

    assert first == second
    assert first["schema_version"] == "canine_identity.source_provenance.v3"
    assert first["entrypoints"] == ["evaluation.splits.registry_cli"]
    paths = [row["relative_path"] for row in first["code_source_files"]]
    assert "enrollment/registry/identity_registry.py" in paths
    assert "shared/foundation/provenance.py" in paths
    assert all(not path.startswith("src/cvi/") for path in paths)

def test_v3_source_closure_binds_parent_package_initializers(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    initializer = package / "__init__.py"
    initializer.write_text("VALUE = 1\n", encoding="utf-8")
    module = package / "module.py"
    module.write_text("VALUE = 2\n", encoding="utf-8")

    with patch("shared.contracts.source_provenance._repository_root", return_value=tmp_path):
        first = build_source_provenance((module,))
        initializer.write_text("VALUE = 3\n", encoding="utf-8")
        second = build_source_provenance((module,))

    assert [row["relative_path"] for row in first["code_source_files"]] == [
        "package/__init__.py",
        "package/module.py",
    ]
    assert first["code_source_manifest_sha256"] != second[
        "code_source_manifest_sha256"
    ]

def test_face_v3_uses_source_closure_and_legacy_hashes_are_narrow() -> None:
    current = build_faceid_source_contract(ROOT, architecture="cls_residual_v5")
    assert current["schema_version"] == (
        "canine_identity.faceid_architecture_input_contract.v3"
    )
    assert current["source_provenance"] == build_source_provenance(
        (
            ROOT / "identification" / "export" / "face" / "residual_model.py",
            ROOT / "identification" / "training" / "face" / "dataset.py",
        ),
        logical_component="embedding.methods.face.cls_residual_v5",
    )

    legacy = expected_faceid_contract_for_checkpoint(
        {"schema_version": "cvi.faceid_architecture_input_contract.v2"},
        ROOT,
        architecture="cls_residual_v5",
    )
    assert legacy["architecture_source_sha256"] == (
        "4801633230047cc837247fd9dabe86304558c72f4480dc3079559293b6ce7611"
    )
