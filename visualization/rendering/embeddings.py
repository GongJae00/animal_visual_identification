"""Caption-free diagnostics for final channel embedding matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from visualization.rendering.style import PAPER_COLORS, paper_matplotlib_rc

_FORMAT = "pdf"
_LABELED_DIMENSIONS = 32
_PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#9D755D",
    "#72B7B2",
    "#D67195",
)
_INK = PAPER_COLORS["ink"]


def render_embedding_views(
    payload: Mapping[str, Any], directory: Path
) -> tuple[list[Path], dict[str, Any]]:
    matrix = _embeddings(payload.get("embeddings"))
    record: dict[str, Any] = {
        "n": 0 if matrix is None else int(matrix.shape[0]),
        "dim": 0 if matrix is None else int(matrix.shape[1]),
        "files": [],
    }
    if matrix is None:
        return [], record

    n = len(matrix)
    groups = _groups(payload, n)
    group_name = "identity" if "identity" in groups else "dataset"
    labels = groups.get(group_name)
    display_labels = _display_labels(labels) if labels is not None else {}
    heatmap_group = (
        "detection"
        if "detection" in groups
        else "dataset"
        if "dataset" in groups
        else group_name
    )
    heatmap_labels = groups.get(heatmap_group)
    heatmap_display_labels = (
        _display_labels(heatmap_labels) if heatmap_labels is not None else {}
    )
    norms = np.linalg.norm(matrix, axis=1)
    backbone_id = payload.get("backbone_id")
    if isinstance(backbone_id, str) and backbone_id:
        record["backbone_id"] = backbone_id
    record["norm_mean"] = float(np.mean(norms))
    record["norm_std"] = float(np.std(norms))
    if labels is not None:
        record["group"] = group_name
        record["group_counts"] = _counts(labels)
        record["group_labels"] = {
            display: original for original, display in display_labels.items()
        }

    order = _row_order(groups, n, primary=heatmap_group)
    ordered = matrix[order]
    ordered_heatmap_labels = (
        tuple(heatmap_display_labels[heatmap_labels[index]] for index in order)
        if heatmap_labels is not None
        else None
    )
    important_dimensions = _important_dimensions(matrix, k=_LABELED_DIMENSIONS)
    labeled_dimensions = _tick_dimensions(important_dimensions, matrix.shape[1])
    written = _save_heatmap(
        directory / "embedding_heatmap",
        ordered,
        labels=ordered_heatmap_labels,
        dimensions=labeled_dimensions,
    )
    record["heatmap_rows"] = len(ordered)
    record["heatmap_group"] = heatmap_group
    if heatmap_labels is not None:
        record["heatmap_group_counts"] = _counts(heatmap_labels)
    record["heatmap_dimensions"] = list(range(matrix.shape[1]))
    record["heatmap_labeled_dimensions"] = [
        int(index) for index in labeled_dimensions
    ]
    record["heatmap_dimension_scores"] = [
        float(value)
        for value in np.mean(np.abs(matrix), axis=0)[labeled_dimensions]
    ]

    points, explained, components = _pca(matrix)
    record["pca_explained"] = [float(value) for value in explained[:3]]
    record["pca_variance_components"] = len(explained)
    record["pca_components_top"] = _top_component_loadings(components, k=8)
    pca_labeled_dimensions = _component_dimensions(components, k=_LABELED_DIMENSIONS)
    record["pca_component_dimensions"] = list(range(matrix.shape[1]))
    record["pca_component_labeled_dimensions"] = [
        int(index) for index in _tick_dimensions(pca_labeled_dimensions, matrix.shape[1])
    ]
    written.extend(_save_pca_variance(directory / "pca_variance", explained))
    written.extend(
        _save_pca_components(
            directory / "pca_components",
            components,
            dimensions=pca_labeled_dimensions,
        )
    )

    if labels is not None and points.shape[1] >= 2:
        scatter_labels = tuple(
            heatmap_display_labels[heatmap_labels[index]]
            for index in range(len(points))
        )
        written.extend(
            _save_pca_identity(
                directory / "pca_identity",
                points[:, :2],
                scatter_labels,
                explained[:2],
            )
        )
        record["pca_group"] = heatmap_group
        record["pca_group_counts"] = _counts(heatmap_labels)

    record["files"] = [path.name for path in written]
    return written, record


def _embeddings(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("embeddings must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("embeddings must be a non-empty 2-d matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain non-finite values")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    return matrix


def _groups(payload: Mapping[str, Any], n: int) -> dict[str, tuple[str, ...]]:
    source = payload.get("groups")
    source = source if isinstance(source, Mapping) else payload
    aliases = {
        "identity": ("identity", "dog_id"),
        "dataset": ("dataset", "domain", "source"),
        "detection": ("detection", "detection_status", "parser_detection"),
        "view": ("view", "viewpoint", "pose"),
    }
    groups: dict[str, tuple[str, ...]] = {}
    for name, keys in aliases.items():
        raw = next((source[key] for key in keys if key in source), None)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) != n:
            raise ValueError(f"{name} labels must be a list of length {n}")
        groups[name] = tuple(str(item) for item in raw)
    return groups


def _display_labels(labels: Sequence[str] | None) -> dict[str, str]:
    if labels is None:
        return {}
    unique = sorted(set(labels))
    if len(unique) <= 8:
        return {label: label for label in unique}
    width = 8
    while len({label[:width] for label in unique}) != len(unique):
        width += 1
    return {label: label[:width] for label in unique}


def _counts(labels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _row_order(
    groups: Mapping[str, Sequence[str]], n: int, *, primary: str
) -> np.ndarray:
    if primary not in groups:
        return np.arange(n)
    rank = {
        "detected_samples": 0,
        "undetected_samples": 1,
    }
    return np.asarray(
        sorted(
            range(n),
            key=lambda index: (
                rank.get(groups[primary][index], 2)
                if primary == "detection"
                else 0,
                groups[primary][index],
                index,
            ),
        )
    )


def _important_dimensions(matrix: np.ndarray, *, k: int) -> np.ndarray:
    scores = np.mean(np.abs(matrix), axis=0)
    return np.argsort(-scores, kind="stable")[: min(k, matrix.shape[1])]


def _component_dimensions(components: np.ndarray, *, k: int) -> np.ndarray:
    scores = np.max(np.abs(components), axis=0)
    return np.argsort(-scores, kind="stable")[: min(k, components.shape[1])]


def _tick_dimensions(dimensions: Sequence[int], width: int) -> np.ndarray:
    candidates = np.unique(
        np.concatenate((np.asarray([0], dtype=int), np.asarray(dimensions, dtype=int)))
    )
    gap = max(1, width // 64)
    return candidates[np.r_[True, np.diff(candidates) >= gap]]


def _pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = matrix - np.mean(matrix, axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    count = min(3, components.shape[0])
    points = centered @ components[:count].T
    variance = singular * singular / max(len(matrix) - 1, 1)
    total = float(np.sum(variance))
    explained = variance / total if total > 0.0 else np.zeros_like(variance)
    return points, explained, components[:count]


def _top_component_loadings(
    components: np.ndarray, *, k: int
) -> list[list[dict[str, float | int]]]:
    rows: list[list[dict[str, float | int]]] = []
    for component in components:
        indices = np.argsort(-np.abs(component))[: min(k, len(component))]
        rows.append(
            [
                {"dimension": int(index), "loading": float(component[index])}
                for index in indices
            ]
        )
    return rows


def _save_heatmap(
    stem: Path,
    matrix: np.ndarray,
    labels: Sequence[str] | None,
    *,
    dimensions: Sequence[int],
) -> list[Path]:
    return _render(
        stem,
        figsize=(8.0, 5.4),
        drawer=lambda figure: _draw_heatmap(figure, matrix, labels, dimensions),
    )


def _save_pca_variance(stem: Path, explained: np.ndarray) -> list[Path]:
    return _render(
        stem,
        figsize=(6.4, 3.4),
        drawer=lambda figure: _draw_pca_variance(figure, explained),
    )


def _save_pca_components(
    stem: Path, components: np.ndarray, *, dimensions: Sequence[int]
) -> list[Path]:
    return _render(
        stem,
        figsize=(7.2, 3.6),
        drawer=lambda figure: _draw_pca_components(figure, components, dimensions),
    )


def _save_pca_identity(
    stem: Path,
    points: np.ndarray,
    labels: Sequence[str],
    explained: np.ndarray,
) -> list[Path]:
    return _render(
        stem,
        figsize=(9.0, 5.2),
        drawer=lambda figure: _draw_pca_identity(figure, points, labels, explained),
    )


def _draw_heatmap(
    figure: Any,
    matrix: np.ndarray,
    labels: Sequence[str] | None,
    dimensions: Sequence[int],
) -> None:
    ax = figure.add_axes((0.10, 0.13, 0.78, 0.78))
    limit = max(float(np.quantile(np.abs(matrix), 0.995)), 1e-8)
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("sample")
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    tick_dimensions = _tick_dimensions(dimensions, matrix.shape[1])
    ax.set_xticks(
        tick_dimensions,
        [f"D{dimension:03d}" for dimension in tick_dimensions],
        rotation=90,
        ha="center",
    )
    ax.tick_params(axis="x", labelbottom=True, labelsize=6)
    if labels is not None:
        centers, names = _run_centers(labels)
        stride = max(1, (len(names) + 19) // 20)
        ax.set_yticks(centers[::stride], names[::stride])
        ax.tick_params(axis="y", labelleft=True)
        for boundary in _run_boundaries(labels):
            ax.axhline(boundary - 0.5, color="white", linewidth=0.35)
    else:
        ax.set_yticks([])
    cax = figure.add_axes((0.91, 0.13, 0.025, 0.78))
    colorbar = figure.colorbar(image, cax=cax)
    colorbar.set_label("embedding value")
    colorbar.outline.set_linewidth(0.4)


def _draw_pca_variance(figure: Any, explained: np.ndarray) -> None:
    ax = figure.add_axes((0.11, 0.18, 0.84, 0.72))
    x = np.arange(len(explained))
    ax.bar(x, explained, color=_INK, width=0.8, linewidth=0)
    ticks = _sparse_ticks(len(explained), limit=16)
    ax.set_xticks(ticks, [f"PC{index + 1}" for index in ticks], rotation=45, ha="right")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance")
    ax.set_ylim(0.0, max(float(np.max(explained)) * 1.12, 0.01))
    _show_xy_spines(ax)


def _draw_pca_components(
    figure: Any, components: np.ndarray, dimensions: Sequence[int]
) -> None:
    values = components
    ax = figure.add_axes((0.12, 0.18, 0.76, 0.68))
    limit = max(float(np.max(np.abs(values))), 1e-8)
    from matplotlib.colors import TwoSlopeNorm

    image = ax.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
        rasterized=True,
    )
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    tick_dimensions = _tick_dimensions(dimensions, values.shape[1])
    ax.set_xticks(
        tick_dimensions,
        [f"D{item:03d}" for item in tick_dimensions],
        rotation=90,
        ha="center",
    )
    ax.tick_params(axis="x", labelbottom=True, labelsize=6)
    ax.set_yticks(
        np.arange(len(components)),
        [f"PC{index + 1}" for index in range(len(components))],
    )
    ax.tick_params(axis="y", labelleft=True)
    ax.set_xlabel("original embedding dimension")
    ax.set_ylabel("principal component")
    cax = figure.add_axes((0.92, 0.18, 0.025, 0.68))
    colorbar = figure.colorbar(image, cax=cax)
    colorbar.set_label("loading")
    colorbar.outline.set_linewidth(0.4)


def _draw_pca_identity(
    figure: Any,
    points: np.ndarray,
    labels: Sequence[str],
    explained: np.ndarray,
) -> None:
    colors, mapping = _colors(labels)
    ax = figure.add_axes((0.08, 0.15, 0.67, 0.76))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=_marker_size(len(points)),
        c=colors,
        linewidths=0,
        alpha=0.78,
        rasterized=True,
    )
    ax.set_xlabel(f"PC1 ({_percent(float(explained[0]))})")
    ax.set_ylabel(f"PC2 ({_percent(float(explained[1]))})")
    _show_xy_spines(ax)
    _in_axes_legend(ax, mapping)


def _colors(labels: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    unique = tuple(sorted(set(labels)))
    if len(unique) <= len(_PALETTE):
        mapping = {label: _PALETTE[index] for index, label in enumerate(unique)}
    else:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        palette = colormaps["turbo"](np.linspace(0.03, 0.97, len(unique)))
        mapping = {label: to_hex(palette[index]) for index, label in enumerate(unique)}
    return [mapping[label] for label in labels], mapping


def _run_centers(labels: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    centers: list[float] = []
    names: list[str] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            centers.append((start + index - 1) / 2.0)
            names.append(labels[start])
            start = index
    return np.asarray(centers), names


def _run_boundaries(labels: Sequence[str]) -> list[int]:
    return [
        index for index in range(1, len(labels)) if labels[index] != labels[index - 1]
    ]


def _marker_size(count: int) -> float:
    if count > 2_000:
        return 4.0
    if count > 500:
        return 8.0
    return 12.0


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _sparse_ticks(size: int, *, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, num=limit, dtype=int))


def _show_xy_spines(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
        color=_INK,
        labelcolor=_INK,
        width=0.6,
        length=3,
        labelsize=7,
    )


def _in_axes_legend(ax: Any, mapping: Mapping[str, str]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=5,
            label=label,
        )
        for label, color in mapping.items()
    ]
    columns = min(3, max(1, (len(handles) + 13) // 14))
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=6,
        ncol=columns,
        borderaxespad=0.0,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def _render(stem: Path, *, figsize: tuple[float, float], drawer: Any) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    rc = paper_matplotlib_rc()
    target = stem.with_suffix(f".{_FORMAT}")
    with matplotlib.rc_context(rc):
        figure = plt.figure(figsize=figsize, facecolor=rc["figure.facecolor"])
        try:
            drawer(figure)
            figure.savefig(
                target,
                format=_FORMAT,
                dpi=rc["savefig.dpi"],
                facecolor=rc["savefig.facecolor"],
                edgecolor="none",
                bbox_inches="tight",
                pad_inches=0.04,
                metadata={"Creator": "visualization.embeddings.v2"},
            )
        finally:
            plt.close(figure)
    return [target]
