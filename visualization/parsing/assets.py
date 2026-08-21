"""Lazy, cached access to private parsing visualization assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visualization.privacy import validate_relative_asset_path


class AssetLoader:
    """Resolve each asset once and reuse its decoded image during one render."""

    __slots__ = ("_root", "_images")

    def __init__(self, root: Path | None) -> None:
        self._root = root
        self._images: dict[Path, Any] = {}

    def image(self, value: Any) -> Any:
        path = self._resolve(value)
        image = self._images.get(path)
        if image is None:
            image = _read_rgb(path)
            self._images[path] = image
        return image

    def _resolve(self, value: Any) -> Path:
        if self._root is None:
            raise ValueError("visualization samples require --asset-root")
        self._root = self._root.resolve(strict=True)
        relative = validate_relative_asset_path(value)
        path = (self._root / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(self._root):
            raise FileNotFoundError(path)
        return path


def _read_rgb(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    return image
