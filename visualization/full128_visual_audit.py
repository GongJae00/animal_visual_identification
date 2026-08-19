"""Private PNG-only evidence plates for the Full128 successor audit.

This module deliberately does not use the generic visualization publication
system.  It consumes already validated private evidence and writes only the
nine requested PNG plates.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXPECTED_FILENAMES = (
    "01_successor_inputs.png",
    "02_auxiliary_inputs.png",
    "03_terminal_inputs.png",
    "04_gallery_k1_schema.png",
    "05_retrieval_high.png",
    "06_retrieval_middle.png",
    "07_retrieval_low.png",
    "08_embedding_distribution.png",
    "09_b5_executed_spatial_trace.png",
)


@dataclass(frozen=True, slots=True)
class AuditSample:
    token: str
    identity: str | None
    dataset: str
    rgb: np.ndarray
    mask: np.ndarray
    route: str = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RankedTemplate:
    token: str
    score: float
    relevant: bool


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    token: str
    cohort_key: tuple[str, str, int]
    relevant_rank: int
    margin: float
    b3_ranked: tuple[RankedTemplate, ...]
    b5_ranked: tuple[RankedTemplate, ...]


def neutralized_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the exact pre-normalization neural RGB input in [0, 1]."""

    values = np.asarray(rgb, dtype=np.float32) / 255.0
    foreground = np.asarray(mask, dtype=bool)
    if values.shape != (224, 224, 3) or foreground.shape != (224, 224):
        raise ValueError("Full128 audit inputs must be 224x224 RGB and mask")
    return values * foreground[..., None] + IMAGENET_MEAN_RGB * (~foreground[..., None])


def normalized_neural_input(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the production ImageNet channel normalization after neutralization."""

    return (neutralized_rgb(rgb, mask) - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB


def select_occupancy_quantiles(
    samples: Sequence[AuditSample],
) -> tuple[AuditSample, ...]:
    """Select low, median, and high occupancy samples with token tie-breaking."""

    if not samples:
        return ()
    ordered = sorted(samples, key=lambda item: (float(item.mask.mean()), item.token))
    indices = (0, (len(ordered) - 1) // 2, len(ordered) - 1)
    return tuple(ordered[index] for index in indices)


def reconstruct_ranking(
    *,
    query_token: str,
    gallery_tokens: Sequence[str],
    vectors: Mapping[str, np.ndarray],
    identities: Mapping[str, str],
) -> tuple[RankedTemplate, ...]:
    """Reconstruct deterministic K=1 template ranking from validated cache vectors."""

    query = np.asarray(vectors[query_token], dtype=np.float32)
    expected_identity = identities[query_token]
    ranked = sorted(
        (
            RankedTemplate(
                token=token,
                score=float(
                    np.dot(query, np.asarray(vectors[token], dtype=np.float32))
                ),
                relevant=identities[token] == expected_identity,
            )
            for token in gallery_tokens
        ),
        key=lambda item: (-item.score, item.token),
    )
    if not any(item.relevant for item in ranked):
        raise ValueError("closed-set audit query lacks a relevant gallery template")
    return tuple(ranked)


def relevant_rank(ranked: Sequence[RankedTemplate]) -> int:
    for index, item in enumerate(ranked, start=1):
        if item.relevant:
            return index
    raise ValueError("ranking lacks relevant template")


def select_outcome_strata(rows: Sequence[QueryOutcome]) -> dict[str, QueryOutcome]:
    """Choose the prescribed high, middle, and low B5 outcome rows."""

    if not rows:
        raise ValueError("DEV K=1 audit has no query rows")
    high = min(rows, key=lambda item: (-item.margin, item.token))
    middle_order = sorted(
        rows, key=lambda item: (item.relevant_rank, -item.margin, item.token)
    )
    low = min(rows, key=lambda item: (-item.relevant_rank, item.token))
    return {"high": high, "middle": middle_order[len(middle_order) // 2], "low": low}


def render_png_audit(
    *,
    output_dir: Path,
    input_lanes: Mapping[str, Sequence[tuple[str, Sequence[AuditSample]]]],
    gallery_query: QueryOutcome,
    outcomes: Mapping[str, QueryOutcome],
    samples: Mapping[str, AuditSample],
    dev_population: Mapping[str, tuple[Sequence[str], Sequence[str]]],
    b3_vectors: Mapping[str, np.ndarray],
    b5_vectors: Mapping[str, np.ndarray],
    trace: Mapping[str, Any],
) -> None:
    """Write all requested plates atomically into a prepared empty directory."""

    _require_empty_png_target(output_dir)
    pyplot = _pyplot()
    _render_input_plate(
        pyplot,
        output_dir / EXPECTED_FILENAMES[0],
        input_lanes["successor"],
        "successor inputs",
    )
    _render_input_plate(
        pyplot,
        output_dir / EXPECTED_FILENAMES[1],
        input_lanes["auxiliary"],
        "auxiliary inputs",
    )
    _render_input_plate(
        pyplot,
        output_dir / EXPECTED_FILENAMES[2],
        input_lanes["terminal"],
        "terminal inputs",
    )
    _render_gallery_plate(
        pyplot, output_dir / EXPECTED_FILENAMES[3], gallery_query, samples
    )
    for filename, stratum in zip(
        EXPECTED_FILENAMES[4:7], ("high", "middle", "low"), strict=True
    ):
        _render_retrieval_plate(
            pyplot, output_dir / filename, stratum, outcomes[stratum], samples
        )
    _render_embedding_plate(
        pyplot,
        output_dir / EXPECTED_FILENAMES[7],
        dev_population,
        b3_vectors,
        b5_vectors,
        outcomes,
    )
    _render_trace_plate(pyplot, output_dir / EXPECTED_FILENAMES[8], trace, samples)
    observed = tuple(sorted(path.name for path in output_dir.iterdir()))
    if observed != EXPECTED_FILENAMES:
        raise RuntimeError("Full128 audit output differs from the required PNG set")


def _require_empty_png_target(output_dir: Path) -> None:
    if not output_dir.is_dir() or output_dir.is_symlink() or any(output_dir.iterdir()):
        raise FileExistsError("Full128 audit output directory must exist and be empty")


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    return pyplot


def _render_input_plate(
    pyplot: Any,
    path: Path,
    datasets: Sequence[tuple[str, Sequence[AuditSample]]],
    title: str,
) -> None:
    rows = len(datasets)
    figure, axes = pyplot.subplots(
        rows, 9, figsize=(18, max(3.2, rows * 3.0)), squeeze=False
    )
    quantiles = ("low occupancy", "median occupancy", "high occupancy")
    for row, (dataset, selected) in enumerate(datasets):
        for quantile in range(3):
            axis_group = axes[row, quantile * 3 : quantile * 3 + 3]
            if quantile >= len(selected):
                _unavailable(axis_group, f"{dataset}\nunavailable")
                continue
            _sample_triplet(
                axis_group, selected[quantile], f"{dataset}\n{quantiles[quantile]}"
            )
    figure.suptitle(
        f"Full128 {title}: raw RGB, route-specific mask, neutral RGB", fontsize=14
    )
    figure.text(
        0.5,
        0.01,
        "neutral RGB = RGB x mask + ImageNet mean x (1-mask), before normalization",
        ha="center",
        fontsize=8,
    )
    _save_png(figure, path)


def _mask_title(route: str) -> str:
    return {
        "NATIVE_FACE": "source-frame validity\n(no segmentation)",
        "NATIVE_HEAD": "source-frame validity\n(no segmentation)",
        "BODY_PARSING": "parser foreground mask",
        "BODY_MASK": "authoritative foreground mask",
    }.get(route, "binary mask")


def _sample_triplet(axes: Sequence[Any], sample: AuditSample, label: str) -> None:
    neutral = neutralized_rgb(sample.rgb, sample.mask)
    mask_title = _mask_title(sample.route)
    for axis, image, title in zip(
        axes,
        (sample.rgb, sample.mask, neutral),
        ("raw RGB", mask_title, "neutral RGB"),
        strict=True,
    ):
        axis.imshow(
            image,
            cmap="gray" if image.ndim == 2 else None,
            vmin=0 if image.ndim == 2 else None,
            vmax=1 if image.ndim == 2 else None,
        )
        axis.set_title(title, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].set_ylabel(label, fontsize=7)


def _unavailable(axes: Sequence[Any], label: str) -> None:
    for axis in axes:
        axis.set_facecolor("#eeeeee")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#bb2222")
    axes[1].text(0.5, 0.5, label, ha="center", va="center", fontsize=8, color="#882222")


def _render_gallery_plate(
    pyplot: Any, path: Path, outcome: QueryOutcome, samples: Mapping[str, AuditSample]
) -> None:
    figure, axes = pyplot.subplots(2, 5, figsize=(15, 6))
    _sample_triplet(axes[0, :3], samples[outcome.token], "DEV query")
    for axis in axes[0, 3:]:
        axis.axis("off")
    _ranked_images(axes[1], outcome.b5_ranked[:5], samples, "B5-SPATIAL K=1")
    figure.suptitle(
        f"DEV K=1 gallery/query evidence | gallery identities/templates: {len(outcome.b5_ranked)} / {len(outcome.b5_ranked)}",
        fontsize=13,
    )
    _save_png(figure, path)


def _render_retrieval_plate(
    pyplot: Any,
    path: Path,
    stratum: str,
    outcome: QueryOutcome,
    samples: Mapping[str, AuditSample],
) -> None:
    figure, axes = pyplot.subplots(3, 5, figsize=(15, 9))
    _sample_triplet(axes[0, :3], samples[outcome.token], f"DEV {stratum} outcome")
    for axis in axes[0, 3:]:
        axis.axis("off")
    _ranked_images(axes[1], outcome.b3_ranked[:5], samples, "B3 top-5")
    _ranked_images(axes[2], outcome.b5_ranked[:5], samples, "B5-SPATIAL top-5")
    figure.suptitle(
        f"DEV {stratum} outcome | B5 relevant rank {outcome.relevant_rank} | B5 top-1 margin {outcome.margin:.6f}",
        fontsize=13,
    )
    _save_png(figure, path)


def _ranked_images(
    axes: Sequence[Any],
    ranked: Sequence[RankedTemplate],
    samples: Mapping[str, AuditSample],
    label: str,
) -> None:
    for index, axis in enumerate(axes):
        if index >= len(ranked):
            axis.axis("off")
            continue
        item = ranked[index]
        axis.imshow(samples[item.token].rgb)
        axis.set_title(
            f"{label if index == 0 else ''}\nrank {index + 1} | {item.score:.6f}\n{'relevant' if item.relevant else 'not relevant'}",
            fontsize=7,
        )
        axis.set_xticks([])
        axis.set_yticks([])


def _render_embedding_plate(
    pyplot: Any,
    path: Path,
    dev_population: Mapping[str, tuple[Sequence[str], Sequence[str]]],
    b3_vectors: Mapping[str, np.ndarray],
    b5_vectors: Mapping[str, np.ndarray],
    outcomes: Mapping[str, QueryOutcome],
) -> None:
    all_queries = sorted(
        {token for query, _ in dev_population.values() for token in query}
    )
    all_gallery = sorted(
        {token for _, gallery in dev_population.values() for token in gallery}
    )
    tokens = tuple(sorted(set(all_queries) | set(all_gallery)))
    figure, axes = pyplot.subplots(1, 2, figsize=(14, 6))
    for axis, vectors, label in zip(
        axes, (b3_vectors, b5_vectors), ("B3", "B5-SPATIAL"), strict=True
    ):
        points = _pca2(np.stack([vectors[token] for token in tokens]))
        position = dict(zip(tokens, points, strict=True))
        axis.scatter(
            *np.asarray([position[token] for token in all_gallery]).T,
            s=10,
            c="#5f7ea8",
            label="gallery",
        )
        axis.scatter(
            *np.asarray([position[token] for token in all_queries]).T,
            s=10,
            c="#d68a4a",
            label="query",
        )
        for name, outcome in outcomes.items():
            winner = outcome.b5_ranked[0].token
            source, destination = position[outcome.token], position[winner]
            axis.plot(
                (source[0], destination[0]),
                (source[1], destination[1]),
                color="#333333",
                alpha=0.45,
                linewidth=0.7,
            )
            axis.scatter(
                source[0], source[1], s=35, marker="x", label=f"{name} selected"
            )
        axis.set_title(f"{label}: DEV K=1 cached 128D vectors")
        axis.set_xlabel("PCA 1")
        axis.set_ylabel("PCA 2")
        axis.legend(fontsize=7, loc="best")
    figure.suptitle(
        "Query/gallery roles; lines join selected queries to their B5 rank-1 template",
        fontsize=12,
    )
    _save_png(figure, path)


def _pca2(matrix: np.ndarray) -> np.ndarray:
    centered = np.asarray(matrix, dtype=np.float64) - np.mean(
        matrix, axis=0, keepdims=True
    )
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return centered @ right[:2].T


def _render_trace_plate(
    pyplot: Any,
    path: Path,
    trace: Mapping[str, Any],
    samples: Mapping[str, AuditSample],
) -> None:
    private = trace["private_samples"]
    maps = trace["available_maps"]
    figure, axes = pyplot.subplots(2, 6, figsize=(18, 6))
    for row, role in enumerate(("query", "key")):
        sample = samples[private[f"{role}_sample_token"]]
        visual = (sample.rgb, sample.mask, neutralized_rgb(sample.rgb, sample.mask))
        labels = ("raw RGB", "binary mask", "neutral RGB")
        for column, (image, label) in enumerate(zip(visual, labels, strict=True)):
            axes[row, column].imshow(
                image,
                cmap="gray" if image.ndim == 2 else None,
                vmin=0 if image.ndim == 2 else None,
                vmax=1 if image.ndim == 2 else None,
            )
            axes[row, column].set_title(label, fontsize=8)
        map_names = (
            f"{role}_pooling_weight",
            f"{role}_spatial_scorer_logit",
            f"{role}_pair_contribution",
        )
        map_labels = (
            "16x16 pooling weight",
            "16x16 scorer logit",
            "16x16 pair contribution",
        )
        for column, (name, label) in enumerate(
            zip(map_names, map_labels, strict=True), start=3
        ):
            image = np.asarray(maps[name]["values"], dtype=np.float32)
            axes[row, column].imshow(image, cmap="magma")
            axes[row, column].set_title(label, fontsize=8)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(role, fontsize=9)
    figure.suptitle(
        "Executed B5-SPATIAL trace | pair contribution is affine contribution, not semantic correspondence",
        fontsize=12,
    )
    _save_png(figure, path)


def _save_png(figure: Any, path: Path) -> None:
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    parent = path.parent
    with NamedTemporaryFile(dir=parent.parent, suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, format="png", dpi=160, metadata={})
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        figure.clf()
