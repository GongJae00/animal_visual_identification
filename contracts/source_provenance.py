"""Deterministic source and Python runtime provenance for offline tools."""

from __future__ import annotations

import ast
import hashlib
import os
import platform
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from foundation.provenance import content_sha256


def build_offline_tool_provenance(
    tool_path: Path,
    *,
    additional_paths: Iterable[Path] = (),
    logical_component: str | None = None,
) -> dict[str, Any]:
    """Bind a tool to the recursive closure of repository-local imports."""

    source = build_source_provenance(
        (tool_path, *additional_paths), logical_component=logical_component
    )
    runtime = _runtime_provenance()
    return {
        "schema_version": "canine_identity.source_provenance.v3",
        **{name: value for name, value in source.items() if name != "schema_version"},
        "runtime": runtime,
        "runtime_sha256": content_sha256(runtime),
    }


def build_source_provenance(
    entry_paths: Iterable[Path],
    *,
    logical_component: str | None = None,
) -> dict[str, Any]:
    """Bind a logical component to its recursive repository-local source closure."""

    repository_root = _repository_root(Path(__file__).resolve())
    entries = tuple(
        dict.fromkeys(path.resolve(strict=True) for path in entry_paths)
    )
    if not entries:
        raise ValueError("source provenance requires at least one entry point")
    sources = _source_closure(entries, repository_root)
    rows = [_source_row(path, repository_root) for path in sources]
    entrypoints = sorted(_logical_source_name(path, repository_root) for path in entries)
    component = logical_component or entrypoints[0]
    if not component or component != component.strip():
        raise ValueError("logical_component must be canonical non-empty text")
    return {
        "schema_version": "canine_identity.source_closure.v3",
        "logical_component": component,
        "entrypoints": entrypoints,
        "code_source_manifest_sha256": content_sha256(rows),
        "code_source_files": rows,
    }


def build_legacy_offline_tool_provenance_v1(
    tool_path: Path,
    *,
    additional_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build the historical flat manifest required by existing PDQ artifacts."""

    package_root = Path(__file__).resolve().parent
    repository_root = _repository_root(Path(__file__).resolve())
    resolved_tool = tool_path.resolve(strict=True)
    requested = (
        *sorted(package_root.glob("*.py")),
        resolved_tool,
        *(path.resolve(strict=True) for path in additional_paths),
    )
    sources = tuple(dict.fromkeys(requested))
    rows: list[dict[str, str | int]] = []
    for path in sources:
        resolved = path.resolve(strict=True)
        try:
            label = resolved.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError("offline tool source is outside repository") from error
        if path.is_symlink() or not resolved.is_file():
            raise ValueError("offline tool source must be a real regular file")
        digest, observed = _hash_regular_file_same_read(resolved)
        rows.append(
            {
                "relative_path": label,
                "content_sha256": digest,
                "byte_size": observed,
            }
        )
    rows.sort(key=lambda item: str(item["relative_path"]))
    runtime = _runtime_provenance()
    return {
        "schema_version": "cvi.offline_tool_provenance.v1",
        "code_source_manifest_sha256": content_sha256(rows),
        "code_source_files": rows,
        "runtime": runtime,
        "runtime_sha256": content_sha256(runtime),
    }


def _repository_root(source: Path) -> Path:
    for parent in source.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("source provenance cannot locate the repository root")


def _source_closure(entries: tuple[Path, ...], repository_root: Path) -> tuple[Path, ...]:
    pending = list(entries)
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        resolved = path.resolve(strict=True)
        if resolved in observed:
            continue
        try:
            resolved.relative_to(repository_root)
        except ValueError as error:
            raise ValueError("offline tool source is outside repository") from error
        if resolved.is_symlink() or not resolved.is_file() or resolved.suffix != ".py":
            raise ValueError("offline tool source must be a real Python file")
        observed.add(resolved)
        pending.extend(_parent_package_initializers(resolved, repository_root))
        pending.extend(_local_import_paths(resolved, repository_root))
    return tuple(sorted(observed, key=lambda path: _logical_source_name(path, repository_root)))


def _parent_package_initializers(
    path: Path,
    repository_root: Path,
) -> tuple[Path, ...]:
    """Return package initializers Python executes before importing ``path``."""

    relative_parent = path.relative_to(repository_root).parent
    initializers: list[Path] = []
    current = repository_root
    for part in relative_parent.parts:
        if part == "src" and current == repository_root:
            current /= part
            continue
        current /= part
        initializer = current / "__init__.py"
        if initializer.is_file() and initializer != path:
            initializers.append(initializer)
    return tuple(initializers)


def _local_import_paths(path: Path, repository_root: Path) -> tuple[Path, ...]:
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot parse provenance source {path}") from error
    current = _logical_source_name(path, repository_root).split(".")
    if path.name != "__init__.py":
        current.pop()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = current[: max(0, len(current) - node.level + 1)] if node.level else []
            module = [part for part in (node.module or "").split(".") if part]
            qualified = ".".join((*base, *module))
            if qualified:
                names.add(qualified)
                names.update(
                    f"{qualified}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
    resolved: set[Path] = set()
    for name in names:
        candidate = _resolve_local_module(name, repository_root)
        if candidate is not None:
            resolved.add(candidate)
    return tuple(sorted(resolved))


def _resolve_local_module(name: str, repository_root: Path) -> Path | None:
    relative = Path(*name.split("."))
    for source_root in (repository_root, repository_root / "src"):
        module = source_root / relative.with_suffix(".py")
        package = source_root / relative / "__init__.py"
        if module.is_file():
            return module
        if package.is_file():
            return package
    return None


def _logical_source_name(path: Path, repository_root: Path) -> str:
    relative = path.resolve(strict=True).relative_to(repository_root)
    if relative.parts[0] == "src":
        relative = Path(*relative.parts[1:])
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _source_row(path: Path, repository_root: Path) -> dict[str, str | int]:
    digest, observed = _hash_regular_file_same_read(path)
    return {
        "logical_name": _logical_source_name(path, repository_root),
        "relative_path": path.relative_to(repository_root).as_posix(),
        "content_sha256": digest,
        "byte_size": observed,
    }


def _runtime_provenance() -> dict[str, str]:
    return {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "os_name": os.name,
        "numpy_version": _distribution_version("numpy"),
        "jsonschema_version": _distribution_version("jsonschema"),
        "scikit_learn_version": _distribution_version("scikit-learn"),
    }


def _hash_regular_file_same_read(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    observed = 0
    try:
        initial = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            observed += len(chunk)
        final = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(initial) != identity(final) or identity(initial) != identity(named):
        raise RuntimeError("offline tool source changed while hashing")
    if observed != initial.st_size:
        raise RuntimeError("offline tool source byte count differs")
    return digest.hexdigest(), observed


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"
