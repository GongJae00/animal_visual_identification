"""Ranked retrieval contact-sheet rendering with content-bound local images."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from vis.privacy import validate_relative_asset_path
from vis.style import COLORS

MAX_IMAGE_PIXELS = 40_000_000
MAX_ENCODED_IMAGE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


def draw_ranked_retrieval(
    figure: Any,
    payload: dict[str, Any],
    *,
    asset_root: Path | None,
) -> None:
    if asset_root is None:
        raise ValueError("ranked retrieval rendering requires an asset root")
    root = asset_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    grid = figure.add_gridspec(
        2,
        5,
        width_ratios=(1.28, 1.0, 1.0, 1.0, 1.0),
        wspace=0.22,
        hspace=0.52,
    )
    query_ax = figure.add_subplot(grid[:, 0])
    _draw_card(
        query_ax,
        _load_bound_image(root, payload["query"]),
        header="QUERY",
        footer="Q",
        color=COLORS["blue"],
    )
    for index, item in enumerate(payload["candidates"]):
        row, column = divmod(index, 4)
        ax = figure.add_subplot(grid[row, column + 1])
        relevant = item["outcome"] == "relevant"
        margin = item["margin"]
        footer = "REL" if relevant else "NON"
        if margin is not None:
            footer += f"  m {margin:+.3f}"
        _draw_card(
            ax,
            _load_bound_image(root, item),
            header=f"R{item['rank']}  s {item['score']:.3f}",
            footer=footer,
            color=COLORS["teal"] if relevant else COLORS["red"],
        )
    figure.text(
        0.56,
        0.105,
        "Border: teal = relevant/correct  |  red = non-relevant  |  "
        "s = cosine score  |  m = score - next rank",
        ha="center",
        fontsize=6.5,
        color=COLORS["muted"],
    )


def _draw_card(
    ax: Any,
    image: Image.Image,
    *,
    header: str,
    footer: str,
    color: str,
) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(header, fontsize=7.2, color=COLORS["ink"], pad=4)
    ax.set_xlabel(footer, fontsize=6.5, color=color, labelpad=3, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2.2)


def _load_bound_image(root: Path, item: dict[str, Any]) -> Image.Image:
    relative = validate_relative_asset_path(item["path"])
    path = root / relative
    if path.is_symlink():
        raise ValueError(f"asset must not be a symlink: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"asset escapes asset root: {relative}") from exc

    descriptor = _open_asset_descriptor(root, relative)
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        initial = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"asset must be a regular file: {relative}")
        if initial.st_size > MAX_ENCODED_IMAGE_BYTES:
            raise ValueError(f"asset exceeds encoded byte limit: {relative}")

        digest = hashlib.sha256()
        hashed_bytes = 0
        while True:
            chunk = stream.read(
                min(_READ_CHUNK_BYTES, MAX_ENCODED_IMAGE_BYTES - hashed_bytes + 1)
            )
            if not chunk:
                break
            hashed_bytes += len(chunk)
            if hashed_bytes > MAX_ENCODED_IMAGE_BYTES:
                raise ValueError(f"asset exceeds encoded byte limit: {relative}")
            digest.update(chunk)
        if hashed_bytes != initial.st_size:
            raise RuntimeError(f"asset changed while being hashed: {relative}")
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"asset hash differs: {relative}")

        stream.seek(0)
        with Image.open(stream) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"asset dimensions exceed limits: {relative}")
            source.load()
            image = source.convert("RGB")
        final = os.fstat(stream.fileno())
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ):
            raise RuntimeError(f"asset changed while being decoded: {relative}")
        return image


def _open_asset_descriptor(root: Path, relative: str) -> int:
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)

    directory_descriptor = os.open(root, directory_flags)
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            next_descriptor = os.open(
                part, directory_flags, dir_fd=directory_descriptor
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
            if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
                raise ValueError(f"asset parent must be a directory: {relative}")
        try:
            return os.open(parts[-1], file_flags, dir_fd=directory_descriptor)
        except OSError as exc:
            raise ValueError(f"asset cannot be opened safely: {relative}") from exc
    finally:
        os.close(directory_descriptor)
