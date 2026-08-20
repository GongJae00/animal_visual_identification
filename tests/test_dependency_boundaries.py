from __future__ import annotations

import ast
from pathlib import Path

from tests.repo_root import REPO_ROOT as ROOT

INTERNAL_PACKAGES = {
    "archive",
    "prototype",
    "data",
    "enrollment",
    "evaluation",
    "gallery",
    "identification",
    "operations",
    "parsing",
    "representation",
    "search",
    "shared",
    "visualization",
}
ALGORITHM_PACKAGES = {
    "enrollment",
    "gallery",
    "identification",
    "representation",
    "search",
    "parsing",
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
        imports.update(name for name in names if name.split(".", 1)[0] in INTERNAL_PACKAGES)
    return imports

def test_dependency_direction() -> None:
    missing_packages = [
        package
        for package in sorted(INTERNAL_PACKAGES)
        if not (ROOT / package / "__init__.py").is_file()
    ]
    assert not missing_packages, f"internal packages are absent: {missing_packages}"
    violations: list[str] = []
    for package in sorted(INTERNAL_PACKAGES):
        for path in sorted((ROOT / package).rglob("*.py")):
            imported = _internal_imports(path)
            roots = {name.split(".", 1)[0] for name in imported}
            if package == "shared":
                relative = path.relative_to(ROOT).parts
                other = roots - {"shared"}
                if relative[:2] == ("shared", "foundation"):
                    leaked = {
                        name
                        for name in imported
                        if name.split(".", 1)[0] == "shared"
                        and not (
                            name == "shared.foundation"
                            or name.startswith("shared.foundation.")
                        )
                    }
                    forbidden = other | leaked
                elif relative[:2] == ("shared", "contracts"):
                    leaked = {
                        name
                        for name in imported
                        if name.split(".", 1)[0] == "shared"
                        and not (
                            name == "shared"
                            or name == "shared.foundation"
                            or name.startswith("shared.foundation.")
                            or name == "shared.contracts"
                            or name.startswith("shared.contracts.")
                        )
                    }
                    forbidden = other | leaked
                else:
                    forbidden = other
            elif package == "prototype":
                relative = path.relative_to(ROOT).parts
                if relative[1:2] == ("commands",):
                    forbidden = roots & set()
                elif relative[:2] == ("prototype", "runtime"):
                    forbidden = roots & {
                        "evaluation",
                        "operations",
                        "archive",
                        "visualization",
                        "parsing",
                    }
                    if any(
                        name.startswith("identification.training")
                        for name in imported
                    ):
                        forbidden.add("identification.training")
                else:
                    forbidden = roots & {
                        "operations",
                        "archive",
                        "visualization",
                    }
                    if any(
                        name.startswith("identification.training")
                        or name.startswith("parsing.training")
                        for name in imported
                    ):
                        forbidden.add("training")
            elif package == "data":
                relative = path.relative_to(ROOT).parts
                if relative[1:2] == ("commands",):
                    forbidden = set()
                else:
                    forbidden = roots & {
                        "identification",
                        "representation",
                        "evaluation",
                        "enrollment",
                        "gallery",
                        "parsing",
                        "search",
                        "archive",
                        "operations",
                        "prototype",
                        "visualization",
                    }
            elif package == "evaluation":
                relative = path.relative_to(ROOT).parts
                if relative[1:2] == ("commands",):
                    forbidden = set()
                else:
                    forbidden = roots & {"operations"}
            elif package in ALGORITHM_PACKAGES:
                relative = path.relative_to(ROOT).parts
                if relative[1:2] == ("commands",):
                    forbidden = set()
                else:
                    forbidden = roots & {
                        "evaluation",
                        "operations",
                        "archive",
                        "visualization",
                    }
                if package == "parsing":
                    forbidden.update(
                        roots
                        & {
                            "identification",
                            "representation",
                            "enrollment",
                            "gallery",
                            "search",
                            "prototype",
                            "operations",
                        }
                    )
                    if relative[:2] == ("parsing", "export"):
                        leaked = {
                            name
                            for name in imported
                            if name == "parsing.training"
                            or name.startswith("parsing.training.")
                        }
                        forbidden.update(leaked)
                if package == "identification" and relative[:2] == (
                    "identification",
                    "export",
                ):
                    leaked = {
                        name
                        for name in imported
                        if name == "identification.training"
                        or name.startswith("identification.training.")
                    }
                    forbidden.update(leaked)
                if package == "representation":
                    leaked = {
                        name
                        for name in imported
                        if name.startswith("identification.training")
                        or name.startswith("parsing.training")
                    }
                    forbidden.update(leaked)
            elif package == "visualization":
                leaked = {
                    name
                    for name in imported
                    if name.startswith("identification.training")
                    or name.startswith("parsing.training")
                }
                forbidden = leaked
            else:
                forbidden = set()
            if forbidden:
                violations.append(
                    f"{path.relative_to(ROOT)} imports {sorted(forbidden)}"
                )
    assert not violations, "\n".join(violations)

def test_internal_package_graph_is_acyclic() -> None:
    dependencies: dict[str, set[str]] = {package: set() for package in INTERNAL_PACKAGES}
    for package in sorted(INTERNAL_PACKAGES):
        for path in sorted((ROOT / package).rglob("*.py")):
            relative = path.relative_to(ROOT).parts
            if "commands" in relative:
                continue
            dependencies[package].update(
                name.split(".", 1)[0]
                for name in _internal_imports(path)
                if name.split(".", 1)[0] != package
            )

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(package: str) -> None:
        if package in visiting:
            start = visiting.index(package)
            cycle = visiting[start:] + [package]
            raise AssertionError(f"internal package cycle: {' -> '.join(cycle)}")
        if package in visited:
            return
        visiting.append(package)
        for dependency in sorted(dependencies[package]):
            visit(dependency)
        visiting.pop()
        visited.add(package)

    for package in sorted(INTERNAL_PACKAGES):
        visit(package)

def test_removed_top_level_packages_are_gone() -> None:
    for package in (
        "embedding",
        "identity",
        "retrieval",
        "runtime",
        "systems",
        "workflows",
        "legacy",
    ):
        assert not (ROOT / package / "__init__.py").is_file(), package
    assert (ROOT / "data" / "commands" / "download.py").is_file()
    assert (ROOT / "visualization" / "commands" / "render.py").is_file()
    assert (ROOT / "visualization" / "parsing" / "__init__.py").is_file()
    assert (ROOT / "prototype" / "runtime" / "__init__.py").is_file()

def test_legacy_is_not_top_level() -> None:
    assert not (ROOT / "legacy" / "__init__.py").is_file()
    assert (ROOT / "archive" / "appearance_face_nose" / "__init__.py").is_file()
    assert (ROOT / "archive" / "nose_metric" / "__init__.py").is_file()
    assert (ROOT / "archive" / "shared_helpers" / "__init__.py").is_file()
    assert (ROOT / "evaluation" / "search_metrics" / "__init__.py").is_file()
    assert (ROOT / "evaluation" / "verification" / "__init__.py").is_file()
    assert (ROOT / "evaluation" / "verification" / "metrics.py").is_file()
    assert (ROOT / "data" / "public_sources" / "public_dataset.py").is_file()
    assert not (ROOT / "data" / "public").exists()
    assert (ROOT / "tests" / "repo_root.py").is_file()
    assert (ROOT / "tests" / "parsing" / "test_detection.py").is_file()
    assert (ROOT / "tests" / "archive" / "full128" / "test_full128_training.py").is_file()
    assert not (ROOT / "identification" / "commands" / "onnx.py").is_file()
    assert (ROOT / "identification" / "commands" / "export.py").is_file()
    assert not (ROOT / "parsing" / "commands" / "export_oracle_crops.py").is_file()
    assert not (ROOT / "data" / "commands" / "audit_public_canine_phash.py").is_file()

def test_runtime_and_systems_are_not_top_level() -> None:
    assert not (ROOT / "runtime" / "__init__.py").is_file()
    assert not (ROOT / "systems" / "__init__.py").is_file()
    assert (ROOT / "prototype" / "__init__.py").is_file()
    assert (ROOT / "prototype" / "runtime" / "__init__.py").is_file()
    assert (ROOT / "operations" / "__init__.py").is_file()

def test_identity_and_retrieval_are_not_top_level() -> None:
    assert not (ROOT / "identity" / "__init__.py").is_file()
    assert not (ROOT / "retrieval" / "__init__.py").is_file()
    assert (ROOT / "enrollment" / "__init__.py").is_file()
    assert (ROOT / "gallery" / "__init__.py").is_file()
    assert (ROOT / "search" / "__init__.py").is_file()
    assert (ROOT / "evaluation" / "splits" / "__init__.py").is_file()

def test_embedding_is_not_top_level() -> None:
    assert not (ROOT / "embedding" / "__init__.py").is_file()
    assert (ROOT / "identification" / "__init__.py").is_file()
    assert (ROOT / "representation" / "__init__.py").is_file()
    assert (ROOT / "archive" / "full128" / "__init__.py").is_file()

def test_foundation_and_contracts_are_not_top_level() -> None:
    assert not (ROOT / "foundation" / "__init__.py").is_file()
    assert not (ROOT / "contracts" / "__init__.py").is_file()
    assert (ROOT / "shared" / "__init__.py").is_file()
    assert (ROOT / "shared" / "foundation" / "__init__.py").is_file()
    assert (ROOT / "shared" / "contracts" / "__init__.py").is_file()
