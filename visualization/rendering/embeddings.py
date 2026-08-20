"""Caption-free embedding views.

PCA scatters, same/different cosine, explained variance, per-dimension
contribution, and per-channel cosine gap. Filenames encode the grouping.
Axis labels stay; titles and captions do not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from visualization.rendering.pipeline import write_optimization_catalog
from visualization.rendering.style import PAPER_COLORS, paper_matplotlib_rc

_FORMAT = "png"
_MAX_POINTS = 4_096
_GROUP_KEYS = (
    ("dataset", ("dataset", "domain", "source")),
    ("identity", ("identity", "dog_id")),
    ("view", ("view", "viewpoint", "pose")),
)
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
_MUTED = PAPER_COLORS["muted"]


def render_vector_stage(
    stage: str,
    trace: dict[str, Any],
    output_root: Path,
    *,
    substages: tuple[str, ...],
    vis_dir: str,
    scope: Any = None,
    **_kwargs: Any,
) -> tuple[Path, ...]:
    from visualization.privacy import PublicationScope, validate_publishable_value

    if scope is None:
        scope = PublicationScope.PRIVATE
    if not isinstance(scope, PublicationScope):
        scope = PublicationScope(str(scope))
    catalog = write_optimization_catalog(stage, trace, output_root)
    if catalog:
        return catalog
    declared = trace.get("stage")
    if declared is not None and declared != stage:
        raise ValueError(f"trace stage {declared!r} does not match {stage!r}")
    validate_publishable_value(trace, scope)
    payload = trace.get("substages")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("trace substages must be an object")
    stage_root = output_root / "vis" / vis_dir
    stage_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    channel_rows: list[dict[str, Any]] = []
    for name in substages:
        sub = payload.get(name)
        if sub is None:
            sub = {}
        if not isinstance(sub, dict):
            raise ValueError(f"substage {name} must be an object")
        target = stage_root / name
        target.mkdir(parents=True, exist_ok=True)
        files, record = render_embedding_views(sub, target)
        record = {
            "stage": stage,
            "substage": name,
            **record,
        }
        json_path = target / "trace.json"
        json_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(json_path)
        written.extend(files)
        cosine = record.get("identity_cosine")
        if isinstance(cosine, dict) and cosine.get("available"):
            channel_rows.append(
                {
                    "channel": _channel_label(name),
                    "gap": cosine["gap"],
                    "same_mean": cosine["same_mean"],
                    "different_mean": cosine["different_mean"],
                }
            )
    if len(channel_rows) >= 2:
        written.extend(_save_channel_gap(stage_root / "channel_gap", channel_rows))
        gap_json = stage_root / "channel_gap.json"
        gap_json.write_text(
            json.dumps(
                {
                    "metric": "same_minus_different_cosine",
                    "channels": channel_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(gap_json)
    return tuple(written)


def render_embedding_views(
    payload: Mapping[str, Any], directory: Path
) -> tuple[list[Path], dict[str, Any]]:
    matrix = _embeddings(payload.get("embeddings"))
    channels = _channel_matrices(payload.get("channels"))
    n = int(matrix.shape[0]) if matrix is not None else 0
    if matrix is None and channels:
        n = int(next(iter(channels.values())).shape[0])
    if channels and n:
        mismatched = [
            name for name, item in channels.items() if int(item.shape[0]) != n
        ]
        if mismatched:
            raise ValueError("channel embeddings must share the sample axis")
    record: dict[str, Any] = {
        "n": n,
        "dim": 0 if matrix is None else int(matrix.shape[1]),
        "files": [],
    }
    written: list[Path] = []
    if n == 0:
        return written, record
    groups_full = _groups(payload, n=n, keep=np.arange(n))
    identity_full = groups_full.get("identity")
    if channels and identity_full is not None:
        channel_rows = _channel_gap_rows(channels, identity_full)
        if len(channel_rows) >= 2:
            written.extend(_save_channel_gap(directory / "channel_gap", channel_rows))
            record["channel_gap"] = channel_rows
    if matrix is None:
        record["files"] = [path.name for path in written]
        return written, record
    points, explained = _pca(matrix)
    record["pca_explained"] = [float(value) for value in explained[:3]]
    record["pca_cumulative"] = _cumulative_marks(explained)
    contrib_kind, contrib = _dimension_contribution(matrix, identity_full)
    record["dim_contrib_kind"] = contrib_kind
    record["dim_contrib_top"] = _top_dims(contrib, k=8)
    rng = np.random.default_rng(0)
    if len(matrix) > _MAX_POINTS:
        keep = np.sort(rng.choice(len(matrix), size=_MAX_POINTS, replace=False))
        points = points[keep]
        drawn = matrix[keep]
        record["drawn"] = int(len(keep))
        record["sampled"] = True
    else:
        keep = np.arange(len(matrix))
        drawn = matrix
        record["drawn"] = int(len(matrix))
        record["sampled"] = False
    groups = {
        name: tuple(labels[index] for index in keep)
        for name, labels in groups_full.items()
    }
    record["groups"] = {name: _counts(labels) for name, labels in groups.items()}
    written.extend(_save_pca_var(directory / "pca_var", explained))
    written.extend(
        _save_dim_contrib(directory / "dim_contrib", contrib, kind=contrib_kind)
    )
    if points.shape[1] >= 2:
        written.extend(
            _save_scatter_2d(
                directory / "pca2",
                points[:, :2],
                labels=None,
                explained=explained[:2],
                legend=False,
            )
        )
        for name, labels in groups.items():
            written.extend(
                _save_scatter_2d(
                    directory / f"pca2_{name}",
                    points[:, :2],
                    labels=labels,
                    explained=explained[:2],
                    legend=_legend_allowed(name, labels),
                )
            )
    if points.shape[1] >= 3:
        written.extend(
            _save_scatter_3d(
                directory / "pca3",
                points[:, :3],
                labels=None,
                explained=explained[:3],
                legend=False,
            )
        )
        for name, labels in groups.items():
            written.extend(
                _save_scatter_3d(
                    directory / f"pca3_{name}",
                    points[:, :3],
                    labels=labels,
                    explained=explained[:3],
                    legend=_legend_allowed(name, labels),
                )
            )
    identity = groups.get("identity")
    if identity is not None:
        cosine = _identity_cosine(drawn, identity)
        if cosine is not None:
            written.extend(_save_cosine(directory / "cosine_identity", cosine))
            record["identity_cosine"] = {
                "available": True,
                "same_mean": float(np.mean(cosine["same"])),
                "different_mean": float(np.mean(cosine["different"])),
                "gap": float(np.mean(cosine["same"]) - np.mean(cosine["different"])),
            }
    record["files"] = [path.name for path in written]
    return written, record


def _channel_matrices(value: Any) -> dict[str, np.ndarray]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not value:
        raise ValueError("channels must be a non-empty object of embedding matrices")
    matrices: dict[str, np.ndarray] = {}
    rows: int | None = None
    for name, raw in value.items():
        if not str(name).strip():
            raise ValueError("channel names must be non-empty")
        matrix = _embeddings(raw)
        if matrix is None:
            continue
        if rows is None:
            rows = int(matrix.shape[0])
        elif int(matrix.shape[0]) != rows:
            raise ValueError("channel embeddings must share the sample axis")
        matrices[str(name)] = matrix
    return matrices


def _channel_gap_rows(
    channels: Mapping[str, np.ndarray], identity: Sequence[str]
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, matrix in channels.items():
        cosine = _identity_cosine(matrix, identity)
        if cosine is None:
            continue
        same = float(np.mean(cosine["same"]))
        different = float(np.mean(cosine["different"]))
        rows.append(
            {
                "channel": name,
                "same_mean": same,
                "different_mean": different,
                "gap": same - different,
            }
        )
    return rows


def _channel_label(substage: str) -> str:
    prefix, separator, rest = substage.partition("_")
    if separator and prefix.isdigit() and rest:
        return rest
    return substage


def _cumulative_marks(explained: np.ndarray) -> dict[str, float]:
    cumulative = np.cumsum(explained)
    marks: dict[str, float] = {}
    for count in (2, 8, 16, 32, 64):
        if len(cumulative) >= count:
            marks[str(count)] = float(cumulative[count - 1])
    marks[str(len(cumulative))] = float(cumulative[-1])
    return marks


def _top_dims(values: np.ndarray, *, k: int) -> list[dict[str, float | int]]:
    order = np.argsort(-values)
    picked = order[: min(k, len(order))]
    return [
        {"dimension": int(index), "value": float(values[index])} for index in picked
    ]


def _dimension_contribution(
    matrix: np.ndarray, identity: Sequence[str] | None
) -> tuple[str, np.ndarray]:
    centered = matrix - np.mean(matrix, axis=0)
    if identity is None or len(set(identity)) < 2:
        variance = np.var(centered, axis=0, ddof=1)
        total = float(np.sum(variance))
        share = variance / total if total > 0.0 else variance
        return "variance_share", share
    labels = np.asarray(identity)
    unique = tuple(sorted(set(identity)))
    within = np.zeros(matrix.shape[1], dtype=np.float64)
    eligible = 0
    means: list[np.ndarray] = []
    for label in unique:
        mask = labels == label
        count = int(np.sum(mask))
        if count == 0:
            continue
        group = centered[mask]
        means.append(np.mean(group, axis=0))
        if count >= 2:
            within += (count - 1) * np.var(group, axis=0, ddof=1)
            eligible += count - 1
    if len(means) < 2:
        variance = np.var(centered, axis=0, ddof=1)
        total = float(np.sum(variance))
        share = variance / total if total > 0.0 else variance
        return "variance_share", share
    between = np.var(np.vstack(means), axis=0, ddof=1)
    if eligible > 0:
        within = within / eligible
    score = between / (between + within + 1e-12)
    return "identity_ratio", score


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


def _pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - np.mean(matrix, axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ right[: min(3, right.shape[0])].T
    ev = singular * singular / max(len(matrix) - 1, 1)
    total = float(np.sum(ev))
    ratio = ev / total if total > 0.0 else np.zeros_like(ev)
    return coords, ratio


def _groups(
    payload: Mapping[str, Any], *, n: int, keep: np.ndarray
) -> dict[str, tuple[str, ...]]:
    nested = payload.get("groups")
    source = nested if isinstance(nested, Mapping) else payload
    result: dict[str, tuple[str, ...]] = {}
    for name, aliases in _GROUP_KEYS:
        raw = None
        for alias in aliases:
            if alias in source and source[alias] is not None:
                raw = source[alias]
                break
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) != n:
            raise ValueError(f"{name} labels must be a list of length {n}")
        labels = tuple(str(item) for item in raw)
        result[name] = tuple(labels[index] for index in keep)
    return result


def _counts(labels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _legend_allowed(name: str, labels: Sequence[str]) -> bool:
    unique = len(set(labels))
    if name == "identity":
        return unique <= 8
    return unique <= 10


def _colors(labels: Sequence[str] | None, count: int) -> tuple[list[str], dict[str, str]]:
    if labels is None:
        return [_INK] * count, {}
    unique = tuple(sorted(set(labels)))
    if len(unique) <= len(_PALETTE):
        mapping = {label: _PALETTE[index] for index, label in enumerate(unique)}
    else:
        mapping = {
            label: _PALETTE[
                hashlib.sha256(label.encode("utf-8")).digest()[0] % len(_PALETTE)
            ]
            for label in unique
        }
    return [mapping[label] for label in labels], mapping


def _marker_size(count: int) -> float:
    if count > 2_000:
        return 4.0
    if count > 500:
        return 8.0
    return 12.0


def _percent(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def _save_scatter_2d(
    stem: Path,
    xy: np.ndarray,
    *,
    labels: Sequence[str] | None,
    explained: np.ndarray,
    legend: bool,
) -> list[Path]:
    colors, mapping = _colors(labels, len(xy))
    return _render(
        stem,
        figsize=(3.4, 3.4),
        drawer=lambda figure: _draw_2d(
            figure, xy, colors, mapping, explained, legend=legend
        ),
    )


def _save_scatter_3d(
    stem: Path,
    xyz: np.ndarray,
    *,
    labels: Sequence[str] | None,
    explained: np.ndarray,
    legend: bool,
) -> list[Path]:
    colors, mapping = _colors(labels, len(xyz))
    return _render(
        stem,
        figsize=(3.6, 3.4),
        drawer=lambda figure: _draw_3d(
            figure, xyz, colors, mapping, explained, legend=legend
        ),
    )


def _save_cosine(
    stem: Path, values: dict[str, np.ndarray]
) -> list[Path]:
    return _render(
        stem,
        figsize=(3.4, 2.6),
        drawer=lambda figure: _draw_cosine(figure, values),
    )


def _save_pca_var(stem: Path, explained: np.ndarray) -> list[Path]:
    return _render(
        stem,
        figsize=(3.4, 2.4),
        drawer=lambda figure: _draw_pca_var(figure, explained),
    )


def _save_dim_contrib(
    stem: Path, values: np.ndarray, *, kind: str
) -> list[Path]:
    ylabel = "identity ratio" if kind == "identity_ratio" else "variance share"
    return _render(
        stem,
        figsize=(3.4, 2.4),
        drawer=lambda figure: _draw_line(
            figure, values, xlabel="dimension", ylabel=ylabel
        ),
    )


def _save_channel_gap(
    stem: Path, rows: Sequence[Mapping[str, Any]]
) -> list[Path]:
    height = max(1.4, 0.55 + 0.38 * len(rows))
    return _render(
        stem,
        figsize=(3.4, height),
        drawer=lambda figure: _draw_channel_gap(figure, rows),
    )


def _draw_2d(
    figure: Any,
    xy: np.ndarray,
    colors: list[str],
    mapping: dict[str, str],
    explained: np.ndarray,
    *,
    legend: bool,
) -> None:
    ax = figure.add_axes((0.16, 0.16, 0.80, 0.80))
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=_marker_size(len(xy)),
        c=colors,
        linewidths=0,
        alpha=0.78,
        rasterized=True,
    )
    ax.set_xlabel(f"PC1 ({_percent(float(explained[0]))})")
    ax.set_ylabel(f"PC2 ({_percent(float(explained[1]))})")
    _show_xy_spines(ax)
    if legend and mapping:
        _in_axes_legend(ax, mapping)


def _draw_3d(
    figure: Any,
    xyz: np.ndarray,
    colors: list[str],
    mapping: dict[str, str],
    explained: np.ndarray,
    *,
    legend: bool,
) -> None:
    ax = figure.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=_marker_size(len(xyz)),
        c=colors,
        linewidths=0,
        alpha=0.78,
        depthshade=False,
    )
    ax.view_init(elev=18, azim=-60)
    ax.set_xlabel(f"PC1 ({_percent(float(explained[0]))})")
    ax.set_ylabel(f"PC2 ({_percent(float(explained[1]))})")
    ax.set_zlabel(f"PC3 ({_percent(float(explained[2]))})")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.6)
        axis.pane.set_edgecolor("#DDDDDD")
    if legend and mapping:
        _in_axes_legend(ax, mapping)


def _draw_cosine(figure: Any, values: dict[str, np.ndarray]) -> None:
    ax = figure.add_axes((0.16, 0.18, 0.80, 0.76))
    bins = np.linspace(-1.0, 1.0, 41)
    series = (
        ("same", values["same"], _PALETTE[0]),
        ("different", values["different"], _MUTED),
    )
    for label, sample, color in series:
        if len(sample) == 0:
            continue
        ax.hist(
            sample,
            bins=bins,
            histtype="step",
            density=True,
            color=color,
            linewidth=1.2,
            label=label,
        )
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("cosine")
    ax.set_ylabel("density")
    _show_xy_spines(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=7)


def _draw_pca_var(figure: Any, explained: np.ndarray) -> None:
    ax = figure.add_axes((0.16, 0.18, 0.80, 0.76))
    cumulative = np.cumsum(explained)
    x = np.arange(1, len(explained) + 1)
    ax.plot(x, cumulative, color=_INK, linewidth=1.2)
    ax.set_xlim(1, len(explained))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("PCs")
    ax.set_ylabel("cumulative")
    _show_xy_spines(ax)


def _draw_line(
    figure: Any, values: np.ndarray, *, xlabel: str, ylabel: str
) -> None:
    ax = figure.add_axes((0.16, 0.18, 0.80, 0.76))
    x = np.arange(len(values))
    ax.plot(x, values, color=_INK, linewidth=0.9)
    ax.set_xlim(0, max(len(values) - 1, 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _show_xy_spines(ax)


def _draw_channel_gap(
    figure: Any, rows: Sequence[Mapping[str, Any]]
) -> None:
    ax = figure.add_axes((0.28, 0.22, 0.68, 0.70))
    names = [str(row["channel"]) for row in rows]
    gaps = np.asarray([float(row["gap"]) for row in rows], dtype=np.float64)
    y = np.arange(len(rows))
    ax.barh(y, gaps, height=0.45, color=_INK, edgecolor="none")
    ax.axvline(0.0, color=_MUTED, linewidth=0.6)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("same − different cosine")
    _show_xy_spines(ax)
    ax.tick_params(axis="y", length=0)


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


def _in_axes_legend(ax: Any, mapping: dict[str, str]) -> None:
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
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=7, markerscale=1.0)


def _identity_cosine(
    matrix: np.ndarray, labels: Sequence[str]
) -> dict[str, np.ndarray] | None:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return None
    directions = matrix / norms
    same: list[float] = []
    different: list[float] = []
    rng = np.random.default_rng(0)
    order = np.arange(len(labels))
    if len(order) > 512:
        order = np.sort(rng.choice(len(order), size=512, replace=False))
    ids = np.asarray(labels)
    for i, left in enumerate(order):
        for right in order[i + 1 :]:
            value = float(np.clip(directions[left] @ directions[right], -1.0, 1.0))
            if ids[left] == ids[right]:
                same.append(value)
            else:
                different.append(value)
    if not same or not different:
        return None
    return {
        "same": np.asarray(same, dtype=np.float64),
        "different": np.asarray(different, dtype=np.float64),
    }


def _render(
    stem: Path,
    *,
    figsize: tuple[float, float],
    drawer: Any,
) -> list[Path]:
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
                metadata={"Software": "visualization.embeddings.v1"},
            )
        finally:
            plt.close(figure)
    return [target]
