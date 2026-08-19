from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PACKAGES = {
    "contracts",
    "runtime",
    "data",
    "embedding",
    "evaluation",
    "foundation",
    "legacy",
    "identity",
    "retrieval",
    "parsing",
    "systems",
    "visualization",
    "workflows",
}
ALGORITHM_PACKAGES = {
    "embedding",
    "retrieval",
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
            if package == "foundation":
                forbidden = roots - {"foundation"}
            elif package == "contracts":
                forbidden = roots - {"contracts", "foundation"}
            elif package == "runtime":
                forbidden = roots & {
                    "evaluation",
                    "legacy",
                    "systems",
                    "workflows",
                }
                if any(name.startswith("embedding.learning") for name in imported):
                    forbidden.add("embedding.learning")
            elif package == "data":
                forbidden = roots & {
                    "embedding",
                    "evaluation",
                    "identity",
                    "parsing",
                    "retrieval",
                    "legacy",
                    "systems",
                    "workflows",
                }
            elif package == "identity":
                forbidden = roots & {
                    "embedding",
                    "evaluation",
                    "legacy",
                    "systems",
                    "workflows",
                }
            elif package == "evaluation":
                forbidden = roots & {"systems"}
            elif package in ALGORITHM_PACKAGES:
                forbidden = roots & {"evaluation", "legacy", "systems", "workflows"}
                relative = path.relative_to(ROOT).parts
                if package == "parsing" and roots & {"embedding", "retrieval"}:
                    forbidden.update(roots & {"embedding", "retrieval"})
                if relative[:2] == ("embedding", "evidence") and any(
                    name.startswith(("embedding.methods", "embedding.learning"))
                    for name in imported
                ):
                    forbidden.add("embedding methods/learning")
                elif relative[:2] == ("embedding", "methods") and any(
                    name.startswith("embedding.learning") for name in imported
                ):
                    forbidden.add("embedding.learning")
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
