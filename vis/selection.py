"""Deterministic registry selection with publication-scope enforcement."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from vis.contracts import FigureContractError, FigureData
from vis.privacy import PublicationScope, scope_allows
from vis.registry import FIGURE_BY_ID, FIGURE_REGISTRY


def select_figures(
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
