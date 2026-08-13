from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PACKAGES = {
    "apps",
    "artifact_contracts",
    "canine_identity",
    "data_pipeline",
    "evaluation",
    "evidence_fusion",
    "experiments",
    "foundation",
    "identity_governance",
    "identity_methods",
    "identity_retrieval",
    "localization",
    "operations",
    "representation_learning",
    "vis",
    "workflows",
}
ALGORITHM_PACKAGES = {
    "evidence_fusion",
    "identity_methods",
    "identity_retrieval",
    "localization",
    "representation_learning",
}


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (node.module,)
        else:
            continue
        imports.update(
            name.split(".", 1)[0]
            for name in names
            if name.split(".", 1)[0] in INTERNAL_PACKAGES
        )
    return imports


def test_dependency_direction() -> None:
    violations: list[str] = []
    for package in sorted(INTERNAL_PACKAGES):
        for path in sorted((ROOT / package).rglob("*.py")):
            imported = _internal_imports(path)
            if package == "foundation":
                forbidden = imported - {"foundation"}
            elif package == "artifact_contracts":
                forbidden = imported - {"artifact_contracts", "foundation"}
            elif package == "canine_identity":
                forbidden = imported & {
                    "apps",
                    "evaluation",
                    "experiments",
                    "operations",
                    "representation_learning",
                    "workflows",
                }
            elif package in ALGORITHM_PACKAGES:
                forbidden = imported & {"evaluation", "operations"}
            else:
                forbidden = set()
            if forbidden:
                violations.append(
                    f"{path.relative_to(ROOT)} imports {sorted(forbidden)}"
                )
    assert not violations, "\n".join(violations)
