from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from visualization import contact_sheet


def _save_image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (8, 6), color).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_bound_image_rejects_oversized_encoded_file(tmp_path: Path) -> None:
    path = tmp_path / "oversized.png"
    with path.open("wb") as stream:
        stream.truncate(contact_sheet.MAX_ENCODED_IMAGE_BYTES + 1)

    with pytest.raises(ValueError, match="encoded byte limit"):
        contact_sheet._load_bound_image(
            tmp_path, {"path": path.name, "sha256": "0" * 64}
        )


def test_load_bound_image_rejects_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _save_image(path, (10, 20, 30))

    with pytest.raises(ValueError, match="asset hash differs"):
        contact_sheet._load_bound_image(
            tmp_path, {"path": path.name, "sha256": "0" * 64}
        )


def test_load_bound_image_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    digest = _save_image(target, (10, 20, 30))
    link = tmp_path / "link.png"
    link.symlink_to(target.name)

    with pytest.raises(ValueError, match="must not be a symlink"):
        contact_sheet._load_bound_image(tmp_path, {"path": link.name, "sha256": digest})


def test_load_bound_image_hashes_and_decodes_same_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "image.png"
    original_color = (10, 20, 30)
    digest = _save_image(path, original_color)
    replacement = tmp_path / "replacement.png"
    replacement_color = (200, 210, 220)
    _save_image(replacement, replacement_color)
    pillow_open = contact_sheet.Image.open

    def replace_path_then_open(stream, *args, **kwargs):
        replacement.replace(path)
        return pillow_open(stream, *args, **kwargs)

    monkeypatch.setattr(contact_sheet.Image, "open", replace_path_then_open)

    loaded = contact_sheet._load_bound_image(
        tmp_path, {"path": path.name, "sha256": digest}
    )

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == original_color
    with pillow_open(path) as current:
        assert current.getpixel((0, 0)) == replacement_color
