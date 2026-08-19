"""Locate the repository root from any legacy file."""

from pathlib import Path


def repository_root(start: Path | str | None = None) -> Path:
    here = Path(start or __file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "AGENTS.md"
        ).is_file():
            return candidate
    raise RuntimeError("animal_visual_identification repository root not found")
