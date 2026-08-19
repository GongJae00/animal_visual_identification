"""Paired comparisons for fixed-panel masked multievidence results."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from evaluation.retrieval import identity_clustered_bootstrap_ci
from legacy.version.afn.experiments.sibetan_multievidence import BRANCHES


METRICS = ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")


def compare_methods_to_appearance(
    result: Mapping[str, Any], *, resamples: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    methods = result.get("methods")
    if not isinstance(methods, dict) or BRANCHES[0] not in methods:
        raise ValueError("masked result methods differ")
    appearance_rows = methods[BRANCHES[0]].get("query_rows")
    if not isinstance(appearance_rows, list) or not appearance_rows:
        raise ValueError("masked Appearance rows differ")
    appearance_by_token = {row["sample_token"]: row for row in appearance_rows}
    if len(appearance_by_token) != len(appearance_rows):
        raise ValueError("masked Appearance rows repeat a query")
    comparisons = {}
    for method_index, (method, outcome) in enumerate(methods.items()):
        if method == BRANCHES[0]:
            continue
        rows = outcome.get("query_rows")
        if not isinstance(rows, list):
            raise ValueError("masked method rows differ")
        if not rows:
            comparisons[method] = {
                "paired_query_count": 0, "paired_identity_count": 0,
                "method_metrics": None, "appearance_metrics": None,
                "delta_bootstrap_cis": None,
            }
            continue
        try:
            baseline = [appearance_by_token[row["sample_token"]] for row in rows]
        except KeyError as error:
            raise ValueError("masked method query escapes Appearance cohort") from error
        comparisons[method] = {
            "paired_query_count": len(rows),
            "paired_identity_count": len({row["bootstrap_cluster_id"] for row in rows}),
            "method_metrics": _metrics(rows),
            "appearance_metrics": _metrics(baseline),
            "delta_bootstrap_cis": {
                metric: identity_clustered_bootstrap_ci(
                    [
                        {
                            "bootstrap_cluster_id": row["bootstrap_cluster_id"],
                            "delta": row[metric] - base[metric],
                        }
                        for row, base in zip(rows, baseline, strict=True)
                    ],
                    metric="delta", resamples=resamples,
                    seed=seed + method_index,
                )
                for metric in METRICS
            },
        }
    return comparisons


def _metrics(rows):
    return {
        "query_count": len(rows),
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
    }


__all__ = ["METRICS", "compare_methods_to_appearance"]
