"""Renderer/style fingerprints and content-bound output inventories."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from foundation.provenance import content_sha256
from visualization.contracts import FigureData
from visualization.rendering.style import STYLE_FINGERPRINT

PROVENANCE_SCHEMA = "cvi.visualization.publication.v1"
INVENTORY_SCHEMA = "cvi.visualization.output_inventory.v1"
RENDERER_VERSION = "cvi.vis.renderer.v1"
_RENDERER_FILES = (
    "rendering/renderer.py",
    "rendering/recipes.py",
    "rendering/diagrams.py",
    "rendering/contact_sheet.py",
    "successor_family.py",
    "rendering/style.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def renderer_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(RENDERER_VERSION.encode("ascii"))
    for name in _RENDERER_FILES:
        digest.update(name.encode("ascii"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def build_inventory(root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    entries = []
    for relative in relative_paths:
        path = root / relative
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": INVENTORY_SCHEMA,
        "inventory_scope": "rendered_artifacts_and_index",
        "excluded_metadata_paths": ["output_inventory.json", "provenance.json"],
        "entries": entries,
    }


def build_provenance(
    *,
    figures: Iterable[FigureData],
    target_scope: str,
    inventory: dict[str, Any],
    matplotlib_version: str,
    publication_strategy: str,
) -> dict[str, Any]:
    figure_list = tuple(figures)
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA,
        "target_scope": target_scope,
        "figure_data": [
            {
                "figure_id": figure.figure_id,
                "figure_data_sha256": content_sha256(figure.to_dict()),
                "source_bindings": [
                    binding.to_dict() for binding in figure.source_bindings
                ],
            }
            for figure in figure_list
        ],
        "renderer": {
            "version": RENDERER_VERSION,
            "fingerprint": renderer_fingerprint(),
            "matplotlib_version": matplotlib_version,
        },
        "style": {
            "fingerprint": STYLE_FINGERPRINT,
        },
        "inventory_sha256": content_sha256(inventory),
        "publication_strategy": publication_strategy,
        "privacy": {
            "absolute_paths_permitted": target_scope == "private",
            "private_identifiers_permitted": target_scope == "private",
        },
    }
    return payload
