from __future__ import annotations

from legacy.version.afn.experiments.masked_comparison import compare_methods_to_appearance
from legacy.version.afn.experiments.sibetan_multievidence import BRANCHES


def _row(token: str, identity: str, hit: float):
    rank = 1 if hit else 2
    return {
        "sample_token": token, "identity_token": identity,
        "bootstrap_cluster_id": identity, "rank": rank,
        "Rank-1": hit, "Rank-5": 1.0, "Rank-10": 1.0,
        "reciprocal_rank": 1.0 / rank,
    }


def test_masked_comparison_uses_exact_method_query_cohort() -> None:
    result = {
        "methods": {
            BRANCHES[0]: {"query_rows": [_row("a", "a", 0.0), _row("b", "b", 1.0)]},
            "A0_plus_N3": {"query_rows": [_row("a", "a", 1.0), _row("b", "b", 1.0)]},
            BRANCHES[1]: {"query_rows": []},
        }
    }
    comparison = compare_methods_to_appearance(result, resamples=20, seed=3)
    fused = comparison["A0_plus_N3"]
    assert fused["paired_query_count"] == 2
    assert fused["method_metrics"]["Rank-1"] == 1.0
    assert fused["appearance_metrics"]["Rank-1"] == 0.5
    assert fused["delta_bootstrap_cis"]["Rank-1"]["estimate"] == 0.5
    assert comparison[BRANCHES[1]]["delta_bootstrap_cis"] is None
