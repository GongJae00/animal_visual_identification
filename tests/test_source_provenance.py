from __future__ import annotations

from pathlib import Path

from artifact_contracts.source_provenance import (
    build_offline_tool_provenance,
    build_source_provenance,
)
from identity_methods.face.checkpoint import (
    build_faceid_source_contract,
    expected_faceid_contract_for_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_source_closure_is_recursive_logical_and_deterministic() -> None:
    tool = ROOT / "workflows" / "build_identity_registry.py"
    first = build_offline_tool_provenance(tool)
    second = build_offline_tool_provenance(tool)

    assert first == second
    assert first["schema_version"] == "canine_identity.source_provenance.v2"
    assert first["entrypoints"] == ["workflows.build_identity_registry"]
    paths = [row["relative_path"] for row in first["code_source_files"]]
    assert "identity_governance/identity_registry.py" in paths
    assert "foundation/provenance.py" in paths
    assert all(not path.startswith("src/cvi/") for path in paths)


def test_face_v3_uses_source_closure_and_legacy_hashes_are_narrow() -> None:
    current = build_faceid_source_contract(ROOT, architecture="cls_residual_v5")
    assert current["schema_version"] == (
        "canine_identity.faceid_architecture_input_contract.v3"
    )
    assert current["source_provenance"] == build_source_provenance(
        (
            ROOT / "identity_methods" / "face" / "residual_model.py",
            ROOT / "identity_methods" / "face" / "dataset.py",
        ),
        logical_component="identity_methods.face.cls_residual_v5",
    )

    legacy = expected_faceid_contract_for_checkpoint(
        {"schema_version": "cvi.faceid_architecture_input_contract.v2"},
        ROOT,
        architecture="cls_residual_v5",
    )
    assert legacy["architecture_source_sha256"] == (
        "4801633230047cc837247fd9dabe86304558c72f4480dc3079559293b6ce7611"
    )
