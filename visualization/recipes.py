"""Validation and drawing for all normalized publication recipe kinds."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from visualization.contact_sheet import draw_ranked_retrieval
from visualization.contracts import FigureContractError, FigureData
from visualization.diagrams import draw_architecture, draw_ladder
from visualization.privacy import validate_relative_asset_path
from visualization.style import COLORS, SERIES_COLORS

_SHA256 = re.compile(r"[0-9a-f]{64}")


def draw_recipe(figure: Any, data: FigureData, *, asset_root: Path | None) -> None:
    payload = data.payload
    validator = _VALIDATORS[data.kind]
    validator(payload)
    if data.kind == "ranked_retrieval":
        draw_ranked_retrieval(figure, payload, asset_root=asset_root)
        return
    if data.kind == "embedding_diagnostics":
        _draw_embedding_diagnostics(figure, payload)
        return
    if data.kind == "score_rank_distributions":
        _draw_score_rank_distributions(figure, payload)
        return
    if data.kind == "model_ladder":
        _draw_model_ladder(figure, payload)
        return
    if data.kind == "result_forest" and "absolute_rows" in payload:
        _draw_successor_results(figure, payload)
        return
    ax = figure.subplots()
    if data.kind == "architecture":
        draw_architecture(ax, payload)
    elif data.kind == "ladder":
        draw_ladder(ax, payload)
    elif data.kind == "census":
        _draw_census(ax, payload)
    elif data.kind == "result_forest":
        _draw_result_forest(ax, payload)
    elif data.kind == "cosine_distribution":
        _draw_cosine_distribution(ax, payload)
    elif data.kind == "embedding_spectrum":
        _draw_embedding_spectrum(ax, payload)
    elif data.kind == "pca_projection":
        _draw_pca_projection(ax, payload)
    elif data.kind == "embedding_topology":
        _draw_embedding_topology(ax, payload)
    elif data.kind == "gallery_composition":
        _draw_gallery_composition(ax, payload)
    else:  # pragma: no cover - registry and validator map make this unreachable
        raise FigureContractError(f"unsupported recipe kind: {data.kind}")


def _draw_census(ax: Any, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    labels = [row["label"] for row in rows]
    values = [row["count"] for row in rows]
    colors = [SERIES_COLORS[row["group_index"] % len(SERIES_COLORS)] for row in rows]
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(payload["x_label"])
    ax.set_xlim(0, payload["x_max"])
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:,}", va="center", fontsize=7.5)
    ax.spines[["right", "top"]].set_visible(False)


def _draw_result_forest(ax: Any, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    y = np.arange(len(rows))
    estimates = np.asarray([row["estimate"] for row in rows])
    lower = np.asarray([row["lower"] for row in rows])
    upper = np.asarray([row["upper"] for row in rows])
    errors = np.vstack((estimates - lower, upper - estimates))
    ax.errorbar(
        estimates,
        y,
        xerr=errors,
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["muted"],
        capsize=3,
    )
    if payload["reference"] is not None:
        ax.axvline(payload["reference"], color=COLORS["orange"], linestyle="--")
    ax.set_yticks(y, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(payload["x_min"], payload["x_max"])
    ax.set_xlabel(payload["x_label"])
    ax.spines[["right", "top"]].set_visible(False)


def _draw_cosine_distribution(ax: Any, payload: dict[str, Any]) -> None:
    edges = np.asarray(payload["bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    for index, series in enumerate(payload["series"]):
        counts = np.asarray(series["counts"], dtype=float)
        denominator = counts.sum()
        density = counts / denominator if denominator else counts
        ax.step(
            centers,
            density,
            where="mid",
            label=series["label"],
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
        )
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(payload["x_label"])
    ax.set_ylabel("Proportion")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)


def _draw_embedding_spectrum(ax: Any, payload: dict[str, Any]) -> None:
    components = payload["components"]
    values = payload["values"]
    ax.plot(components, values, marker="o", markersize=3, color=COLORS["purple"])
    ax.set_xlim(payload["x_min"], payload["x_max"])
    ax.set_ylim(payload["y_min"], payload["y_max"])
    if payload["log_y"]:
        ax.set_yscale("log")
    ax.set_xlabel(payload["x_label"])
    ax.set_ylabel(payload["y_label"])
    ax.spines[["right", "top"]].set_visible(False)


def _draw_pca_projection(ax: Any, payload: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for point in payload["points"]:
        groups.setdefault(point["group"], []).append(point)
    for index, (group, points) in enumerate(sorted(groups.items())):
        ax.scatter(
            [point["x"] for point in points],
            [point["y"] for point in points],
            s=18,
            alpha=0.75,
            label=group,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
        )
    ax.set_xlim(*payload["x_limits"])
    ax.set_ylim(*payload["y_limits"])
    ax.set_xlabel(payload["x_label"])
    ax.set_ylabel(payload["y_label"])
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)


def _draw_embedding_topology(ax: Any, payload: dict[str, Any]) -> None:
    nodes = payload["nodes"]
    for edge in payload["edges"]:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        ax.plot(
            [source["x"], target["x"]],
            [source["y"], target["y"]],
            color=COLORS["grid"],
            linewidth=0.8,
            zorder=1,
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        groups.setdefault(node["group"], []).append(node)
    for index, (group, members) in enumerate(sorted(groups.items())):
        ax.scatter(
            [node["x"] for node in members],
            [node["y"] for node in members],
            label=group,
            s=24,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            zorder=2,
        )
    ax.set_xlim(*payload["x_limits"])
    ax.set_ylim(*payload["y_limits"])
    ax.set_xlabel(payload["x_label"])
    ax.set_ylabel(payload["y_label"])
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)


def _draw_gallery_composition(ax: Any, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    labels = [row["label"] for row in rows]
    values = [row["value"] for row in rows]
    colors = [SERIES_COLORS[row["group_index"] % len(SERIES_COLORS)] for row in rows]
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": COLORS["paper"]},
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.92, 0.5))
    ax.text(0, 0, payload["center_label"], ha="center", va="center", fontsize=9)
    ax.set_aspect("equal")


def _draw_embedding_diagnostics(figure: Any, payload: dict[str, Any]) -> None:
    spectrum_ax, cumulative_ax = figure.subplots(1, 2)
    for index, series in enumerate(payload["series"]):
        color = SERIES_COLORS[series["style_index"] % len(SERIES_COLORS)]
        components = np.arange(1, len(series["explained_variance"]) + 1)
        spectrum_ax.plot(
            components,
            series["explained_variance"],
            label=series["label"],
            color=color,
        )
        cumulative_ax.plot(
            components,
            series["cumulative_variance"],
            label=series["label"],
            color=color,
        )
    spectrum_ax.set_xlim(1, payload["component_count"])
    spectrum_ax.set_ylim(0, payload["variance_y_max"])
    spectrum_ax.set_xlabel("PCA component")
    spectrum_ax.set_ylabel("Explained variance ratio")
    cumulative_ax.set_xlim(1, payload["component_count"])
    cumulative_ax.set_ylim(0, 1.0)
    cumulative_ax.set_xlabel("PCA components retained")
    cumulative_ax.set_ylabel("Cumulative explained variance")
    for ax in (spectrum_ax, cumulative_ax):
        ax.spines[["right", "top"]].set_visible(False)
    handles, labels = spectrum_ax.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=min(5, len(labels)),
        fontsize=6.8,
    )


def _draw_model_ladder(figure: Any, payload: dict[str, Any]) -> None:
    from matplotlib.patches import Rectangle

    ax = figure.subplots()
    variants = {item["alias"]: item for item in payload["variants"]}
    maximum_column = max(item["column"] for item in variants.values())
    positions = {
        alias: (
            0.05 + 0.88 * item["column"] / max(maximum_column, 4),
            0.25 + 0.62 * item["row"],
        )
        for alias, item in variants.items()
    }
    for edge in payload["edges"]:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        ax.annotate(
            "",
            xy=positions[edge["target"]],
            xytext=positions[edge["source"]],
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["muted"],
                "linewidth": 1.0,
                "shrinkA": 24,
                "shrinkB": 24,
            },
            zorder=1,
        )
    for alias, item in variants.items():
        color = COLORS["teal"] if item["status"] == "GO" else COLORS["muted"]
        linestyle = "solid" if item["reported"] else "dashed"
        ax.text(
            *positions[alias],
            f"{alias}\n{item['status']}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=COLORS["ink"],
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": COLORS["paper"],
                "edgecolor": color,
                "linewidth": 1.6,
                "linestyle": linestyle,
            },
            zorder=2,
        )
    boundaries = payload["boundaries"]
    width = 0.90 / len(boundaries)
    for index, boundary in enumerate(boundaries):
        x = 0.05 + index * width
        color = (COLORS["teal"], COLORS["orange"], COLORS["muted"])[index]
        ax.add_patch(
            Rectangle(
                (x, 0.025),
                width - 0.01,
                0.11,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.92,
            )
        )
        ax.text(
            x + 0.012,
            0.08,
            f"{boundary['label']} | {boundary['status'].replace('_', ' ')}\n"
            f"{boundary['detail']}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.4,
            color="white" if index < 2 else COLORS["ink"],
            fontweight="bold",
        )
    ax.set_axis_off()


def _draw_successor_results(figure: Any, payload: dict[str, Any]) -> None:
    from matplotlib.lines import Line2D

    absolute_ax, delta_ax = figure.subplots(
        1, 2, gridspec_kw={"width_ratios": (1.08, 0.92)}
    )
    aliases = [item["alias"] for item in payload["alias_legend"]]
    y_by_alias = {alias: index for index, alias in enumerate(aliases)}
    scope_marker = {"DEV": "o", "CAL": "s", "EXPOSED_DIAGNOSTIC": "^"}
    scope_offset = {"DEV": -0.18, "CAL": 0.0, "EXPOSED_DIAGNOSTIC": 0.18}
    for row in payload["absolute_rows"]:
        y = y_by_alias[row["alias"]] + scope_offset[row["scope"]]
        marker = scope_marker[row["scope"]]
        absolute_ax.scatter(row["rank1"], y, color=COLORS["blue"], marker=marker, s=22)
        absolute_ax.scatter(row["mrr"], y, color=COLORS["orange"], marker=marker, s=22)
    absolute_ax.set_yticks(range(len(aliases)), aliases)
    absolute_ax.invert_yaxis()
    absolute_ax.set_xlim(0, 1)
    absolute_ax.set_xlabel("Aggregate metric")
    absolute_ax.set_title("Absolute Rank-1 / MRR", fontsize=9)
    absolute_ax.spines[["right", "top"]].set_visible(False)
    handles = [
        Line2D(
            [], [], color=COLORS["blue"], marker="o", linestyle="none", label="Rank-1"
        ),
        Line2D(
            [], [], color=COLORS["orange"], marker="o", linestyle="none", label="MRR"
        ),
        *[
            Line2D(
                [],
                [],
                color=COLORS["muted"],
                marker=scope_marker[scope],
                linestyle="none",
                label="EXPOSED" if scope == "EXPOSED_DIAGNOSTIC" else scope,
            )
            for scope in ("DEV", "CAL", "EXPOSED_DIAGNOSTIC")
        ],
    ]
    absolute_ax.legend(handles=handles, loc="upper left", fontsize=6.2, ncol=2)

    delta_rows = payload["delta_rows"]
    if delta_rows:
        y = np.arange(len(delta_rows))
        estimates = np.asarray([row["estimate"] for row in delta_rows])
        lower = np.asarray([row["lower"] for row in delta_rows])
        upper = np.asarray([row["upper"] for row in delta_rows])
        colors = [
            COLORS["blue"] if row["metric"] == "Rank-1" else COLORS["orange"]
            for row in delta_rows
        ]
        for index, color in enumerate(colors):
            delta_ax.errorbar(
                estimates[index],
                y[index],
                xerr=[
                    [estimates[index] - lower[index]],
                    [upper[index] - estimates[index]],
                ],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
            )
        delta_ax.set_yticks([])
        limit = payload["delta_limit"]
        for index, row in enumerate(delta_rows):
            delta_ax.text(
                -limit * 0.96,
                y[index],
                f"{row['right_alias']} | {row['metric']}",
                ha="left",
                va="center",
                fontsize=5.8,
                color=COLORS["muted"],
            )
        delta_ax.invert_yaxis()
    else:
        delta_ax.text(
            0.5,
            0.5,
            "No DEV paired interval available",
            transform=delta_ax.transAxes,
            ha="center",
            va="center",
            color=COLORS["muted"],
            fontsize=7,
        )
        delta_ax.set_yticks([])
    limit = payload["delta_limit"]
    delta_ax.axvline(0, color=COLORS["ink"], linestyle="--", linewidth=1)
    delta_ax.set_xlim(-limit, limit)
    delta_ax.set_xlabel("Paired DEV difference (95% interval)")
    delta_ax.set_title(
        f"DEV delta: {payload['selected_alias']} - comparator", fontsize=9
    )
    delta_ax.spines[["right", "top"]].set_visible(False)

    legend = [
        f"{item['alias']} = {item['description']}" for item in payload["alias_legend"]
    ]
    chunks = [legend[index : index + 3] for index in range(0, len(legend), 3)]
    for index, chunk in enumerate(chunks[:3]):
        figure.text(0.06, 0.125 - index * 0.022, "  |  ".join(chunk), fontsize=5.6)


def _draw_score_rank_distributions(figure: Any, payload: dict[str, Any]) -> None:
    cosine = payload["cosine_distribution"]
    if cosine is None:
        rank_ax = figure.subplots()
        cosine_ax = None
    else:
        cosine_ax, rank_ax = figure.subplots(1, 2)
    for index, series in enumerate(payload["rank_series"]):
        rank_ax.plot(
            series["ranks"],
            series["values"],
            marker="o",
            label=series["label"],
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
        )
    rank_ax.set_xlim(1, payload["rank_x_max"])
    rank_ax.set_ylim(0, 1)
    rank_ax.set_xlabel("Rank k")
    rank_ax.set_ylabel("Cumulative match rate")
    rank_ax.set_xticks(payload["rank_ticks"])
    rank_ax.legend(fontsize=7)
    rank_ax.spines[["right", "top"]].set_visible(False)
    if cosine_ax is not None:
        edges = np.asarray(cosine["bin_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        for index, series in enumerate(cosine["series"]):
            counts = np.asarray(series["counts"], dtype=float)
            cosine_ax.step(
                centers,
                counts / counts.sum(),
                where="mid",
                label=series["label"],
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
            )
        cosine_ax.set_xlim(edges[0], edges[-1])
        cosine_ax.set_ylim(bottom=0)
        cosine_ax.set_xlabel("Cosine similarity")
        cosine_ax.set_ylabel("Proportion")
        cosine_ax.legend(fontsize=7)
        cosine_ax.spines[["right", "top"]].set_visible(False)


def _validate_architecture(payload: dict[str, Any]) -> None:
    _exact(payload, {"nodes", "edges"}, "architecture payload")
    nodes = _nonempty_list(payload["nodes"], "architecture nodes", maximum=40)
    for node in nodes:
        _exact(node, {"label", "layer", "group_index"}, "architecture node")
        _text(node["label"], "node label")
        _integer(node["layer"], "node layer", minimum=0, maximum=12)
        _integer(node["group_index"], "node group_index", minimum=0, maximum=100)
    for edge in _list(payload["edges"], "architecture edges", maximum=100):
        _exact(edge, {"source", "target", "label"}, "architecture edge")
        _integer(edge["source"], "edge source", minimum=0, maximum=len(nodes) - 1)
        _integer(edge["target"], "edge target", minimum=0, maximum=len(nodes) - 1)
        _text(edge["label"], "edge label", allow_empty=True)


def _validate_ladder(payload: dict[str, Any]) -> None:
    _exact(payload, {"steps"}, "ladder payload")
    for step in _nonempty_list(payload["steps"], "ladder steps", maximum=12):
        _exact(step, {"label", "detail", "status"}, "ladder step")
        _text(step["label"], "step label")
        _text(step["detail"], "step detail")
        if step["status"] not in {"established", "conditional", "out_of_scope"}:
            raise FigureContractError("unsupported ladder status")


def _validate_census(payload: dict[str, Any]) -> None:
    _exact(payload, {"rows", "x_label", "x_max"}, "census payload")
    rows = _nonempty_list(payload["rows"], "census rows", maximum=40)
    maximum_count = 0
    for row in rows:
        _exact(row, {"label", "count", "group_index"}, "census row")
        _text(row["label"], "census label")
        maximum_count = max(maximum_count, _integer(row["count"], "count", minimum=0))
        _integer(row["group_index"], "group_index", minimum=0, maximum=100)
    _text(payload["x_label"], "x_label")
    x_max = _number(payload["x_max"], "x_max", minimum=0, strict_minimum=True)
    if maximum_count > x_max:
        raise FigureContractError("census x_max must include all counts")


def _validate_result_forest(payload: dict[str, Any]) -> None:
    if "absolute_rows" in payload:
        _validate_successor_results(payload)
        return
    _exact(
        payload,
        {"rows", "x_label", "x_min", "x_max", "reference"},
        "result forest payload",
    )
    x_min = _number(payload["x_min"], "x_min")
    x_max = _number(payload["x_max"], "x_max")
    if x_min >= x_max:
        raise FigureContractError("result forest x scale is invalid")
    reference = payload["reference"]
    if reference is not None:
        _bounded_number(reference, "reference", x_min, x_max)
    _text(payload["x_label"], "x_label")
    for row in _nonempty_list(payload["rows"], "result rows", maximum=50):
        _exact(row, {"label", "estimate", "lower", "upper"}, "result row")
        _text(row["label"], "result label")
        lower = _bounded_number(row["lower"], "lower", x_min, x_max)
        estimate = _bounded_number(row["estimate"], "estimate", x_min, x_max)
        upper = _bounded_number(row["upper"], "upper", x_min, x_max)
        if not lower <= estimate <= upper:
            raise FigureContractError("result interval must contain its estimate")


def _validate_cosine(payload: dict[str, Any]) -> None:
    _exact(payload, {"bin_edges", "series", "x_label"}, "cosine payload")
    edges = _numeric_list(payload["bin_edges"], "bin_edges", minimum_length=3)
    if any(not -1 <= value <= 1 for value in edges) or any(
        left >= right for left, right in pairwise(edges)
    ):
        raise FigureContractError("cosine bin edges must increase within [-1, 1]")
    series = _nonempty_list(payload["series"], "cosine series", maximum=8)
    labels: set[str] = set()
    for item in series:
        _exact(item, {"label", "counts"}, "cosine series")
        label = _text(item["label"], "series label")
        if label in labels:
            raise FigureContractError("cosine series labels must be unique")
        labels.add(label)
        counts = _list(item["counts"], "counts", maximum=len(edges) - 1)
        if len(counts) != len(edges) - 1:
            raise FigureContractError("cosine counts must align with bin edges")
        for count in counts:
            _integer(count, "bin count", minimum=0)
        if not any(counts):
            raise FigureContractError("cosine series must contain observations")
    _text(payload["x_label"], "x_label")


def _validate_spectrum(payload: dict[str, Any]) -> None:
    expected = {
        "components",
        "values",
        "x_label",
        "y_label",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "log_y",
    }
    _exact(payload, expected, "embedding spectrum payload")
    components = _numeric_list(payload["components"], "components", minimum_length=2)
    values = _numeric_list(payload["values"], "values", minimum_length=2)
    if len(components) != len(values):
        raise FigureContractError("spectrum components and values must align")
    if any(left >= right for left, right in pairwise(components)):
        raise FigureContractError("spectrum components must increase")
    x_min = _number(payload["x_min"], "x_min")
    x_max = _number(payload["x_max"], "x_max")
    y_min = _number(payload["y_min"], "y_min")
    y_max = _number(payload["y_max"], "y_max")
    if x_min >= x_max or y_min >= y_max:
        raise FigureContractError("spectrum scales are invalid")
    if components[0] < x_min or components[-1] > x_max:
        raise FigureContractError("spectrum x scale excludes components")
    if min(values) < y_min or max(values) > y_max:
        raise FigureContractError("spectrum y scale excludes values")
    if not isinstance(payload["log_y"], bool):
        raise FigureContractError("log_y must be boolean")
    if payload["log_y"] and (y_min <= 0 or min(values) <= 0):
        raise FigureContractError("log spectrum values and scale must be positive")
    _text(payload["x_label"], "x_label")
    _text(payload["y_label"], "y_label")


def _validate_projection(payload: dict[str, Any]) -> None:
    _validate_xy_payload(payload, "points", "PCA projection")
    for point in payload["points"]:
        _exact(point, {"x", "y", "group"}, "PCA point")
        _number(point["x"], "point x")
        _number(point["y"], "point y")
        _text(point["group"], "point group")


def _validate_topology(payload: dict[str, Any]) -> None:
    _validate_xy_payload(payload, "nodes", "embedding topology", extra={"edges"})
    nodes = payload["nodes"]
    for node in nodes:
        _exact(node, {"x", "y", "group"}, "topology node")
        _number(node["x"], "node x")
        _number(node["y"], "node y")
        _text(node["group"], "node group")
    for edge in _list(payload["edges"], "topology edges", maximum=10_000):
        _exact(edge, {"source", "target"}, "topology edge")
        source = _integer(
            edge["source"], "edge source", minimum=0, maximum=len(nodes) - 1
        )
        target = _integer(
            edge["target"], "edge target", minimum=0, maximum=len(nodes) - 1
        )
        if source == target:
            raise FigureContractError("topology self-edges are not allowed")


def _validate_gallery(payload: dict[str, Any]) -> None:
    _exact(payload, {"rows", "center_label"}, "gallery composition payload")
    total = 0.0
    for row in _nonempty_list(payload["rows"], "gallery rows", maximum=20):
        _exact(row, {"label", "value", "group_index"}, "gallery row")
        _text(row["label"], "gallery label")
        total += _number(row["value"], "gallery value", minimum=0, strict_minimum=True)
        _integer(row["group_index"], "group_index", minimum=0, maximum=100)
    if total <= 0:
        raise FigureContractError("gallery composition must have positive mass")
    _text(payload["center_label"], "center_label")


def _validate_retrieval(payload: dict[str, Any]) -> None:
    _exact(payload, {"query", "candidates"}, "ranked retrieval payload")
    _validate_image(payload["query"], candidate=False)
    candidates = _nonempty_list(
        payload["candidates"], "retrieval candidates", maximum=8
    )
    for expected_rank, item in enumerate(candidates, 1):
        _validate_image(item, candidate=True)
        if item["rank"] != expected_rank:
            raise FigureContractError(
                "retrieval candidates must have consecutive ranks"
            )
        if (item["margin"] is None) != (expected_rank == len(candidates)):
            raise FigureContractError("only the final retrieval rank may omit margin")


def _validate_embedding_diagnostics(payload: dict[str, Any]) -> None:
    _exact(
        payload,
        {"series", "manifest", "component_count", "variance_y_max"},
        "embedding diagnostics payload",
    )
    component_count = _integer(
        payload["component_count"], "component_count", minimum=2, maximum=128
    )
    variance_y_max = _number(
        payload["variance_y_max"],
        "variance_y_max",
        minimum=0,
        maximum=1,
        strict_minimum=True,
    )
    labels: set[str] = set()
    for series in _nonempty_list(payload["series"], "diagnostic series", maximum=12):
        _exact(
            series,
            {
                "label",
                "style_index",
                "sample_count",
                "explained_variance",
                "cumulative_variance",
            },
            "diagnostic series",
        )
        label = _text(series["label"], "diagnostic label")
        if label in labels:
            raise FigureContractError("diagnostic labels must be unique")
        labels.add(label)
        _integer(series["style_index"], "style_index", minimum=0, maximum=100)
        _integer(series["sample_count"], "sample_count", minimum=2)
        explained = _numeric_list(
            series["explained_variance"],
            "explained_variance",
            minimum_length=component_count,
        )
        cumulative = _numeric_list(
            series["cumulative_variance"],
            "cumulative_variance",
            minimum_length=component_count,
        )
        if len(explained) != component_count or len(cumulative) != component_count:
            raise FigureContractError("diagnostic component counts differ")
        if any(not 0 <= value <= 1 for value in (*explained, *cumulative)):
            raise FigureContractError("diagnostic variance values must be in [0, 1]")
        if any(left > right for left, right in pairwise(cumulative)):
            raise FigureContractError("cumulative variance must not decrease")
        if max(explained) > variance_y_max:
            raise FigureContractError("variance_y_max excludes a spectrum value")
    manifest_aliases = set()
    displayed_aliases = set()
    for item in _nonempty_list(payload["manifest"], "cache manifest", maximum=12):
        _exact(
            item,
            {
                "alias",
                "description",
                "sample_count",
                "cache_descriptor_sha256",
                "displayed",
            },
            "cache manifest row",
        )
        alias = _text(item["alias"], "manifest alias")
        if alias in manifest_aliases:
            raise FigureContractError("cache manifest aliases must be unique")
        manifest_aliases.add(alias)
        _text(item["description"], "manifest description")
        _integer(item["sample_count"], "manifest sample_count", minimum=2)
        if not isinstance(item["displayed"], bool):
            raise FigureContractError("manifest displayed must be boolean")
        if item["displayed"]:
            displayed_aliases.add(alias)
        if not isinstance(
            item["cache_descriptor_sha256"], str
        ) or not _SHA256.fullmatch(item["cache_descriptor_sha256"]):
            raise FigureContractError("manifest cache digest must be lowercase SHA-256")
    if labels != displayed_aliases or not labels <= manifest_aliases:
        raise FigureContractError("displayed diagnostic series and manifest differ")


def _validate_model_ladder(payload: dict[str, Any]) -> None:
    _exact(payload, {"variants", "edges", "boundaries"}, "model ladder payload")
    aliases = set()
    for item in _nonempty_list(payload["variants"], "model variants", maximum=12):
        _exact(
            item,
            {
                "alias",
                "description",
                "status",
                "reported",
                "column",
                "row",
            },
            "model variant",
        )
        alias = _text(item["alias"], "variant alias")
        if alias in aliases:
            raise FigureContractError("model variant aliases must be unique")
        aliases.add(alias)
        _text(item["description"], "variant description")
        if item["status"] not in {"GO", "NO_GO"}:
            raise FigureContractError("model variant status differs")
        if not isinstance(item["reported"], bool):
            raise FigureContractError("model variant reported must be boolean")
        _integer(item["column"], "variant column", minimum=0, maximum=6)
        _number(item["row"], "variant row", minimum=0, maximum=1)
    for edge in _list(payload["edges"], "model edges", maximum=30):
        _exact(edge, {"source", "target"}, "model edge")
        if edge["source"] not in aliases or edge["target"] not in aliases:
            raise FigureContractError("model edge references an unknown alias")
    boundaries = _nonempty_list(payload["boundaries"], "evidence boundaries", maximum=3)
    if len(boundaries) != 3:
        raise FigureContractError("model ladder requires three evidence boundaries")
    for boundary in boundaries:
        _exact(boundary, {"label", "detail", "status"}, "evidence boundary")
        for key in ("label", "detail", "status"):
            _text(boundary[key], f"boundary {key}")


def _validate_successor_results(payload: dict[str, Any]) -> None:
    _exact(
        payload,
        {
            "absolute_rows",
            "delta_rows",
            "alias_legend",
            "selected_alias",
            "delta_limit",
        },
        "successor results payload",
    )
    aliases = set()
    for item in _nonempty_list(payload["alias_legend"], "alias legend", maximum=12):
        _exact(item, {"alias", "description"}, "alias legend row")
        alias = _text(item["alias"], "alias legend alias")
        if alias in aliases:
            raise FigureContractError("result aliases must be unique")
        aliases.add(alias)
        _text(item["description"], "alias legend description")
    if payload["selected_alias"] not in aliases:
        raise FigureContractError("selected publication alias is absent")
    for row in _nonempty_list(payload["absolute_rows"], "absolute rows", maximum=40):
        _exact(row, {"alias", "scope", "rank1", "mrr"}, "absolute result row")
        if row["alias"] not in aliases or row["scope"] not in {
            "DEV",
            "CAL",
            "EXPOSED_DIAGNOSTIC",
        }:
            raise FigureContractError("absolute result alias or scope differs")
        _number(row["rank1"], "absolute Rank-1", minimum=0, maximum=1)
        _number(row["mrr"], "absolute MRR", minimum=0, maximum=1)
    for row in _list(payload["delta_rows"], "delta rows", maximum=24):
        _exact(
            row,
            {
                "left_alias",
                "right_alias",
                "metric",
                "estimate",
                "lower",
                "upper",
            },
            "delta result row",
        )
        if (
            row["left_alias"] not in aliases
            or row["right_alias"] not in aliases
            or row["left_alias"] == row["right_alias"]
            or row["metric"] not in {"Rank-1", "MRR"}
        ):
            raise FigureContractError("delta result aliases or metric differ")
        lower = _number(row["lower"], "delta lower", minimum=-1, maximum=1)
        estimate = _number(row["estimate"], "delta estimate", minimum=-1, maximum=1)
        upper = _number(row["upper"], "delta upper", minimum=-1, maximum=1)
        if not lower <= estimate <= upper:
            raise FigureContractError("delta interval does not contain estimate")
    _number(
        payload["delta_limit"], "delta_limit", minimum=0, maximum=1, strict_minimum=True
    )


def _validate_score_rank_distributions(payload: dict[str, Any]) -> None:
    _exact(
        payload,
        {
            "rank_series",
            "rank_ticks",
            "rank_x_max",
            "cosine_distribution",
        },
        "score/rank distributions payload",
    )
    rank_x_max = _integer(payload["rank_x_max"], "rank_x_max", minimum=1)
    ticks = _numeric_list(payload["rank_ticks"], "rank_ticks", minimum_length=1)
    if any(value < 1 or value > rank_x_max for value in ticks):
        raise FigureContractError("rank ticks are outside the rank scale")
    labels: set[str] = set()
    for series in _nonempty_list(payload["rank_series"], "rank series", maximum=12):
        _exact(series, {"label", "ranks", "values"}, "rank series")
        label = _text(series["label"], "rank label")
        if label in labels:
            raise FigureContractError("rank series labels must be unique")
        labels.add(label)
        ranks = _numeric_list(series["ranks"], "ranks", minimum_length=1)
        values = _numeric_list(series["values"], "rank values", minimum_length=1)
        if (
            len(ranks) != len(values)
            or any(rank < 1 or rank > rank_x_max for rank in ranks)
            or any(left >= right for left, right in pairwise(ranks))
            or any(not 0 <= value <= 1 for value in values)
            or any(left > right for left, right in pairwise(values))
        ):
            raise FigureContractError("rank curve differs")
    cosine = payload["cosine_distribution"]
    if cosine is not None:
        _validate_cosine(cosine)


def _validate_image(value: Any, *, candidate: bool) -> None:
    item = _mapping(value, "image item")
    expected = {"path", "sha256", "label"}
    if candidate:
        expected |= {"rank", "score", "margin", "outcome"}
    _exact(item, expected, "image item")
    validate_relative_asset_path(item["path"])
    if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
        raise FigureContractError("image sha256 must be lowercase SHA-256")
    _text(item["label"], "image label")
    if candidate:
        _integer(item["rank"], "candidate rank", minimum=1, maximum=8)
        _number(item["score"], "candidate score", minimum=-1, maximum=1)
        if item["margin"] is not None:
            _number(item["margin"], "candidate margin", minimum=0, maximum=2)
        if item["outcome"] not in {"relevant", "not_relevant", "unknown"}:
            raise FigureContractError("unsupported retrieval outcome")


def _validate_xy_payload(
    payload: dict[str, Any],
    item_key: str,
    name: str,
    *,
    extra: set[str] | None = None,
) -> None:
    expected = {item_key, "x_limits", "y_limits", "x_label", "y_label"}
    _exact(payload, expected | (extra or set()), f"{name} payload")
    _nonempty_list(payload[item_key], item_key, maximum=10_000)
    for axis in ("x", "y"):
        limits = _numeric_list(
            payload[f"{axis}_limits"], f"{axis}_limits", minimum_length=2
        )
        if len(limits) != 2 or limits[0] >= limits[1]:
            raise FigureContractError(f"{name} {axis} limits are invalid")
        _text(payload[f"{axis}_label"], f"{axis}_label")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureContractError(f"{name} must be an object")
    return dict(value)


def _exact(value: Any, expected: set[str], name: str) -> None:
    item = _mapping(value, name)
    if set(item) != expected:
        raise FigureContractError(f"{name} fields differ")


def _list(value: Any, name: str, *, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise FigureContractError(f"{name} must be an array of at most {maximum} items")
    return value


def _nonempty_list(value: Any, name: str, *, maximum: int) -> list[Any]:
    result = _list(value, name, maximum=maximum)
    if not result:
        raise FigureContractError(f"{name} must not be empty")
    return result


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 160
        or (not allow_empty and not value.strip())
    ):
        raise FigureContractError(f"{name} must be text of at most 160 characters")
    return value


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FigureContractError(f"{name} must be finite")
    if minimum is not None and (
        result <= minimum if strict_minimum else result < minimum
    ):
        raise FigureContractError(f"{name} is below its allowed range")
    if maximum is not None and result > maximum:
        raise FigureContractError(f"{name} is above its allowed range")
    return result


def _bounded_number(value: Any, name: str, lower: float, upper: float) -> float:
    return _number(value, name, minimum=lower, maximum=upper)


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FigureContractError(f"{name} must be an integer of at least {minimum}")
    if maximum is not None and value > maximum:
        raise FigureContractError(f"{name} exceeds {maximum}")
    return value


def _numeric_list(value: Any, name: str, *, minimum_length: int) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) < minimum_length
        or len(value) > 10_000
    ):
        raise FigureContractError(f"{name} has invalid length")
    return [_number(item, name) for item in value]


_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "architecture": _validate_architecture,
    "ladder": _validate_ladder,
    "census": _validate_census,
    "result_forest": _validate_result_forest,
    "cosine_distribution": _validate_cosine,
    "embedding_spectrum": _validate_spectrum,
    "pca_projection": _validate_projection,
    "embedding_topology": _validate_topology,
    "gallery_composition": _validate_gallery,
    "ranked_retrieval": _validate_retrieval,
    "embedding_diagnostics": _validate_embedding_diagnostics,
    "score_rank_distributions": _validate_score_rank_distributions,
    "model_ladder": _validate_model_ladder,
}


def validate_recipe(data: FigureData) -> None:
    try:
        validator = _VALIDATORS[data.kind]
    except KeyError as exc:
        raise FigureContractError(f"unsupported recipe kind: {data.kind}") from exc
    validator(data.payload)
