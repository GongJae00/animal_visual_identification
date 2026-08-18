"""Atomic no-overwrite publication of static visualization bundles."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from visualization.contracts import FigureContractError, FigureData
from visualization.privacy import PublicationScope, scope_allows
from visualization.publishing.provenance import build_inventory, build_provenance
from visualization.publishing.static_index import build_static_index
from visualization.registry import FIGURE_BY_ID, FIGURE_REGISTRY
from visualization.rendering.renderer import matplotlib_version, render_static_figure


def publish(
    figures: tuple[FigureData, ...],
    target: Path,
    *,
    target_scope: PublicationScope,
    asset_root: Path | None = None,
    figure_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Render and atomically publish one complete no-overwrite directory."""

    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite publication: {target}")
    parent = target.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    selected = _select_figures(
        figures,
        target_scope=target_scope,
        figure_ids=figure_ids,
    )
    with TemporaryDirectory(prefix=".cvi-vis-", dir=parent) as temporary:
        staging = Path(temporary) / "publication"
        staging.mkdir(mode=0o700)
        artifact_paths: list[str] = []
        for figure in selected:
            artifact_paths.extend(
                render_static_figure(figure, staging, asset_root=asset_root)
            )
        index_path = staging / "index.html"
        index_path.write_text(
            build_static_index(selected, target_scope=target_scope.value),
            encoding="utf-8",
        )
        artifact_paths.append("index.html")
        inventory = build_inventory(staging, artifact_paths)
        _write_json(staging / "output_inventory.json", inventory)
        provenance = build_provenance(
            figures=selected,
            target_scope=target_scope.value,
            inventory=inventory,
            matplotlib_version=matplotlib_version(),
            publication_strategy="ATOMIC_DIRECTORY_NOREPLACE",
        )
        _write_json(staging / "provenance.json", provenance)
        _fsync_tree(staging)
        strategy = rename_directory_noreplace(staging, parent / target.name)
    fsync_directory(parent / target.name)
    fsync_directory(parent)
    return {
        "target": target.name,
        "figure_ids": [figure.figure_id for figure in selected],
        "inventory_sha256": provenance["inventory_sha256"],
        "publication_strategy": strategy,
    }


def _select_figures(
    figures: Iterable[FigureData],
    *,
    target_scope: PublicationScope,
    figure_ids: Sequence[str] | None = None,
) -> tuple[FigureData, ...]:
    """Validate and order supplied figures according to the canonical registry."""

    supplied: dict[str, FigureData] = {}
    for figure in figures:
        if not isinstance(figure, FigureData):
            raise TypeError("figures must contain FigureData values")
        spec = FIGURE_BY_ID.get(figure.figure_id)
        if spec is None:
            raise FigureContractError(f"unregistered figure_id: {figure.figure_id}")
        if figure.kind != spec.kind and figure.kind not in spec.alternate_kinds:
            raise FigureContractError(
                f"figure kind differs for {figure.figure_id}: expected one of "
                f"{(spec.kind, *spec.alternate_kinds)}"
            )
        if figure.figure_id in supplied:
            raise FigureContractError(f"duplicate figure input: {figure.figure_id}")
        if not scope_allows(target_scope, figure.scope):
            raise PermissionError(
                f"{figure.figure_id} scope {figure.scope.value} exceeds "
                f"publication scope {target_scope.value}"
            )
        supplied[figure.figure_id] = figure

    requested = set(supplied) if figure_ids is None else set(figure_ids)
    if figure_ids is not None and len(requested) != len(figure_ids):
        raise FigureContractError("requested figure IDs must be unique")
    unknown = requested - set(FIGURE_BY_ID)
    if unknown:
        raise FigureContractError(f"unregistered requested figures: {sorted(unknown)}")
    missing = requested - set(supplied)
    if missing:
        raise FigureContractError(
            f"requested figure inputs are missing: {sorted(missing)}"
        )
    selected = tuple(
        supplied[spec.figure_id]
        for spec in FIGURE_REGISTRY
        if spec.figure_id in requested
    )
    if not selected:
        raise FigureContractError("at least one figure must be selected")
    return selected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        fsync_directory(path)
    fsync_directory(root)
