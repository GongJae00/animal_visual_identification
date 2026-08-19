from __future__ import annotations

import numpy as np
import pytest

from legacy.version.afn.experiments.sibetan_multievidence import (
    BRANCHES,
    evaluate_effective_k_panel,
    evaluate_fixed_panel,
    evaluate_n4_substitution,
    face_reliability,
    fit_effective_k_weights,
    nose_reliability,
)
from legacy.version.afn.workflows.evaluate_sibetan_multievidence import (
    _adapt_nose_embeddings,
    _validate_n4_runtime_bindings,
)


FUSION_WEIGHTS = {
    "A0_plus_F0": {BRANCHES[0]: 0.75, BRANCHES[1]: 0.25},
    "A0_plus_N3": {BRANCHES[0]: 1.0, BRANCHES[2]: 0.0},
    "F0_plus_N3": {BRANCHES[1]: 0.75, BRANCHES[2]: 0.25},
    "A0_plus_F0_plus_N3": {
        BRANCHES[0]: 0.75,
        BRANCHES[1]: 0.25,
        BRANCHES[2]: 0.0,
    },
}


def _panel():
    gallery = [
        {"sample_token": "g-a", "identity_token": "a"},
        {"sample_token": "g-b", "identity_token": "b"},
    ]
    queries = [
        {"sample_token": "q-a", "identity_token": "a"},
        {"sample_token": "q-b", "identity_token": "b"},
    ]
    return gallery, queries


def test_fixed_panel_evaluates_available_branches_and_fusions() -> None:
    gallery, queries = _panel()
    vectors = {
        "g-a": np.array([1.0, 0.0]),
        "g-b": np.array([0.0, 1.0]),
        "q-a": np.array([1.0, 0.1]),
        "q-b": np.array([0.1, 1.0]),
    }
    result = evaluate_fixed_panel(
        gallery=gallery,
        queries=queries,
        embeddings={branch: vectors for branch in BRANCHES},
        transfer_weights=FUSION_WEIGHTS,
    )

    assert result["fixed_population"] == {
        "query_count": 2,
        "gallery_template_count": 2,
        "gallery_identity_count": 2,
        "shot": 1,
    }
    assert all(
        outcome["metrics"]["Rank-1"] == 1.0
        for outcome in result["methods"].values()
    )


def test_fixed_panel_abstains_instead_of_shrinking_gallery() -> None:
    gallery, queries = _panel()
    vectors = {
        "g-a": np.array([1.0, 0.0]),
        "g-b": np.array([0.0, 1.0]),
        "q-a": np.array([1.0, 0.1]),
        "q-b": np.array([0.1, 1.0]),
    }
    result = evaluate_fixed_panel(
        gallery=gallery,
        queries=queries,
        embeddings={
            BRANCHES[0]: vectors,
            BRANCHES[1]: {key: value for key, value in vectors.items() if key != "g-b"},
            BRANCHES[2]: vectors,
        },
        transfer_weights=FUSION_WEIGHTS,
    )

    face = result["methods"][BRANCHES[1]]
    assert face["complete_gallery_identity_count"] == 1
    assert face["evaluated_query_count"] == 0
    assert face["abstained_query_count"] == 2
    assert face["metrics"] is None
    assert result["methods"][BRANCHES[0]]["evaluated_query_count"] == 2
    assert result["methods"][BRANCHES[2]]["evaluated_query_count"] == 2


def test_effective_k_fusion_keeps_candidates_and_renormalizes_available_branches() -> None:
    gallery = [
        {"sample_token": "a1", "identity_token": "a"},
        {"sample_token": "a2", "identity_token": "a"},
        {"sample_token": "a3", "identity_token": "a"},
        {"sample_token": "b1", "identity_token": "b"},
        {"sample_token": "b2", "identity_token": "b"},
        {"sample_token": "b3", "identity_token": "b"},
        {"sample_token": "c1", "identity_token": "c"},
        {"sample_token": "c2", "identity_token": "c"},
        {"sample_token": "c3", "identity_token": "c"},
    ]
    queries = [
        {"sample_token": "qa", "identity_token": "a"},
        {"sample_token": "qb", "identity_token": "b"},
        {"sample_token": "qc", "identity_token": "c"},
    ]
    appearance = {
        "a1": np.array([1.0, 0.0, 0.0]), "a2": np.array([1.0, 0.1, 0.0]),
        "a3": np.array([1.0, -0.1, 0.0]), "b1": np.array([0.0, 1.0, 0.0]),
        "b2": np.array([0.1, 1.0, 0.0]), "b3": np.array([-0.1, 1.0, 0.0]),
        "c1": np.array([0.0, 0.0, 1.0]), "c2": np.array([0.0, 0.1, 1.0]),
        "c3": np.array([0.0, -0.1, 1.0]), "qa": np.array([1.0, 0.0, 0.0]),
        "qb": np.array([0.0, 1.0, 0.0]), "qc": np.array([0.0, 0.0, 1.0]),
    }
    face = {
        "a1": np.array([1.0, 0.0, 0.0]), "a3": np.array([1.0, 0.1, 0.0]),
        "b1": np.array([0.0, 1.0, 0.0]),
        "qa": np.array([1.0, 0.0, 0.0]), "qb": np.array([0.0, 1.0, 0.0]),
        "qc": np.array([0.2, 0.8, 0.0]),
    }
    result = evaluate_effective_k_panel(
        gallery=gallery,
        queries=queries,
        embeddings={BRANCHES[0]: appearance, BRANCHES[1]: face, BRANCHES[2]: {}},
        transfer_weights=FUSION_WEIGHTS,
    )

    assert result["branch_availability"][BRANCHES[1]]["gallery_effective_k"] == {"a": 2, "b": 1, "c": 0}
    assert result["methods"][BRANCHES[1]]["evaluated_query_count"] == 0
    fused = result["methods"]["A0_plus_F0"]
    assert fused["evaluated_query_count"] == 3
    assert fused["metrics"]["Rank-1"] == 1.0
    assert fused["candidate_active_branch_patterns"] == {
        BRANCHES[0]: 3,
        f"{BRANCHES[0]}+{BRANCHES[1]}": 6,
    }


def test_effective_k_reports_zero_weight_fusion_as_equivalent() -> None:
    gallery, queries = _panel()
    vectors = {
        "g-a": np.array([1.0, 0.0]), "g-b": np.array([0.0, 1.0]),
        "q-a": np.array([1.0, 0.1]), "q-b": np.array([0.1, 1.0]),
    }
    result = evaluate_effective_k_panel(
        gallery=gallery,
        queries=queries,
        embeddings={BRANCHES[0]: vectors, BRANCHES[1]: {}, BRANCHES[2]: {}},
        transfer_weights=FUSION_WEIGHTS,
    )
    method = result["methods"]["A0_plus_N3"]
    assert method["equivalent_method"] == BRANCHES[0]
    assert method["evaluated_query_count"] == 0


def test_masked_weight_fit_is_development_only_and_deterministic() -> None:
    gallery, queries = _panel()
    vectors = {
        "g-a": np.array([1.0, 0.0]), "g-b": np.array([0.0, 1.0]),
        "q-a": np.array([1.0, 0.1]), "q-b": np.array([0.1, 1.0]),
    }
    result = fit_effective_k_weights(
        gallery=gallery,
        queries=queries,
        embeddings={branch: vectors for branch in BRANCHES},
        resolution=2,
    )
    assert result["labels_used"] == "DEVELOPMENT_ONLY"
    assert result["fusions"]["A0_plus_F0"]["candidate_count"] == 3
    assert result["fusions"]["A0_plus_F0_plus_N3"]["candidate_count"] == 6
    assert sum(result["fusions"]["A0_plus_N3"]["selected_weights"].values()) == 1.0


def test_continuous_reliability_downweights_tiny_profile_without_rejecting() -> None:
    strong = nose_reliability(
        detector_confidence=0.9, frontality=0.8, native_short_side=64,
        blur_score=0.9, contrast_score=0.9,
    )
    weak = nose_reliability(
        detector_confidence=0.9, frontality=0.0, native_short_side=6,
        blur_score=0.2, contrast_score=0.4,
    )
    assert 0.0 < weak < strong <= 1.0
    assert 0.0 < face_reliability(upstream_overall=0.2, native_short_side=8)
    assert face_reliability(upstream_overall=0.8, native_short_side=128) == 0.8


def test_n4_substitution_preserves_availability_and_reports_paired_delta() -> None:
    gallery, queries = _panel()
    baseline = {
        "g-a": np.array([1.0, 0.0]), "g-b": np.array([0.7, 0.7]),
        "q-a": np.array([0.8, 0.2]), "q-b": np.array([1.0, 0.0]),
    }
    adapted = {
        "g-a": np.array([1.0, 0.0]), "g-b": np.array([0.0, 1.0]),
        "q-a": np.array([1.0, 0.0]), "q-b": np.array([0.0, 1.0]),
    }
    embeddings = {branch: baseline for branch in BRANCHES}
    quality = {branch: {token: 1.0 for token in baseline} for branch in BRANCHES}

    result = evaluate_n4_substitution(
        gallery=gallery,
        queries=queries,
        embeddings=embeddings,
        adapted_nose_embeddings=adapted,
        transfer_weights=FUSION_WEIGHTS,
        quality=quality,
        bootstrap_resamples=10,
        bootstrap_seed=3,
    )

    nose = result["methods"][BRANCHES[2]]
    assert result["availability_quality_and_frozen_weights_unchanged"] is True
    assert nose["baseline_N3_metrics"]["Rank-1"] == 0.5
    assert nose["candidate_N4_metrics"]["Rank-1"] == 1.0
    assert nose["rescue_break"]["rescue_count"] == 1
    assert nose["rescue_break"]["break_count"] == 0
    assert nose["paired_N4_minus_N3_bootstrap_cis"]["Rank-1"]["estimate"] == 0.5


def test_n4_substitution_rejects_changed_availability() -> None:
    gallery, queries = _panel()
    vectors = {
        "g-a": np.array([1.0, 0.0]), "g-b": np.array([0.0, 1.0]),
        "q-a": np.array([1.0, 0.1]), "q-b": np.array([0.1, 1.0]),
    }
    embeddings = {branch: vectors for branch in BRANCHES}
    quality = {branch: {token: 1.0 for token in vectors} for branch in BRANCHES}

    with pytest.raises(ValueError, match="exact N3 availability"):
        evaluate_n4_substitution(
            gallery=gallery,
            queries=queries,
            embeddings=embeddings,
            adapted_nose_embeddings={key: value for key, value in vectors.items() if key != "q-b"},
            transfer_weights=FUSION_WEIGHTS,
            quality=quality,
            bootstrap_resamples=2,
            bootstrap_seed=0,
        )


def test_n4_runtime_binding_and_empty_availability_fail_closed() -> None:
    checkpoint = {
        "bindings": {
            "n3": {
                "onnx_sha256": "1" * 64,
                "runtime_manifest_payload_sha256": "2" * 64,
            }
        }
    }
    _validate_n4_runtime_bindings(
        checkpoint, runtime_manifest_sha256="2" * 64, onnx_sha256="1" * 64
    )
    with pytest.raises(ValueError, match="preprocessing"):
        _validate_n4_runtime_bindings(
            checkpoint, runtime_manifest_sha256="3" * 64, onnx_sha256="1" * 64
        )
    assert _adapt_nose_embeddings(checkpoint, {}) == {}
