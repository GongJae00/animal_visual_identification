from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from evaluation.full128_successors import sanitize_successor_evaluation_report
from foundation.provenance import content_sha256
from vis.contracts import FigureContractError
from vis.privacy import PublicationScope
from vis.publication import publish
from vis.successor_family import adapt_successor_family

_ACTUAL_SUCCESSOR_IDS = tuple(
    sorted(
        (
            "B0-FV",
            "B1-FV",
            "B2-FV",
            "B3",
            "B4-U0",
            "B4-U1",
            "B5-UNIFORM",
            "B5-CHANNEL",
            "B5-SPATIAL",
        )
    )
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _scope(successor_id: str, scope: str, rank1: float) -> dict:
    body = {
        "successor_id": successor_id,
        "scope": scope,
        "status": "AVAILABLE",
        "reason": None,
        "query_count": 6,
        "identity_count": 3,
        "metrics": {
            "Rank-1": rank1,
            "Rank-5": min(1.0, rank1 + 0.15),
            "Rank-10": min(1.0, rank1 + 0.2),
            "MRR": min(1.0, rank1 + 0.08),
        },
    }
    return {**body, "result_sha256": content_sha256(body)}


def _selection(candidates: list[dict], selected: str) -> dict:
    rows = [
        {
            "successor_id": candidate["successor_id"],
            "result_sha256": candidate["scope_aggregates"][0]["result_sha256"],
            "objective_value": candidate["scope_aggregates"][0]["metrics"]["Rank-1"],
            "denominator": 6,
        }
        for candidate in candidates
    ]
    body = {
        "schema_version": "cvi.full128_successor_dev_selection_receipt.v1",
        "selection_scope": "DEV_ONLY",
        "objective_metric": "Rank-1",
        "tie_policy": "SUCCESSOR_ID_ASC",
        "candidates": rows,
        "selected_successor_id": selected,
        "calibration_scope_used": False,
        "exposed_scope_used": False,
    }
    return {**body, "receipt_sha256": content_sha256(body)}


def _paired(left: str, right: str, *, scope: str = "DEV") -> dict:
    intervals = []
    for index, metric in enumerate(("Rank-1", "Rank-5", "Rank-10", "MRR")):
        estimate = 0.1 - index * 0.01
        body = {
            "schema_version": "cvi.full128_successor_paired_bootstrap.v1",
            "metric": metric,
            "estimate": estimate,
            "lower_bound": estimate - 0.05,
            "upper_bound": estimate + 0.05,
            "confidence_level": 0.95,
            "cluster_unit": "registered_identity_id",
            "cluster_count": 3,
            "paired_query_count": 6,
            "resamples": 100,
            "seed": 7,
        }
        intervals.append({**body, "bootstrap_sha256": content_sha256(body)})
    return {
        "scope": scope,
        "left_successor_id": left,
        "right_successor_id": right,
        "intervals": intervals,
    }


def _gallery(scope: str, k: int) -> dict:
    return {
        "scope": scope,
        "dataset_name": "dogfacenet224",
        "enrollment_k": k,
        "gallery_sha256": _sha(f"gallery:{scope}:{k}"),
        "scorer_hash": _sha(f"scorer:{scope}:{k}"),
        "template_count": 3 * k,
        "identity_count": 3,
    }


def _public_report(
    *,
    cache_hashes: dict[str, str] | None = None,
    candidate_ids: tuple[str, ...] = ("S0", "S1"),
    selected_id: str = "S1",
) -> dict:
    candidate_ids = tuple(sorted(candidate_ids))
    cache_hashes = cache_hashes or {
        successor_id: _sha(f"cache:{successor_id}") for successor_id in candidate_ids
    }
    candidates = []
    for index, successor_id in enumerate(candidate_ids):
        base = 0.86 if successor_id == selected_id else 0.62 + index * 0.02
        candidates.append(
            {
                "successor_id": successor_id,
                "cache_descriptor_sha256": cache_hashes[successor_id],
                "scope_aggregates": [
                    _scope(successor_id, scope, base - index * 0.05)
                    for index, scope in enumerate(("DEV", "CAL", "EXPOSED_DIAGNOSTIC"))
                ],
                "gallery_bindings": [
                    _gallery(scope, k)
                    for scope in ("DEV", "CAL", "EXPOSED_DIAGNOSTIC")
                    for k in (1, 3, 5)
                ],
            }
        )
    body = {
        "schema_version": "cvi.full128_successor_public_evaluation.v1",
        "visibility": "PUBLIC_AGGREGATE",
        "source_private_report_sha256": _sha("private-report"),
        "evaluation_panel_sha256": _sha("evaluation-panel"),
        "candidates": candidates,
        "dev_selection_receipt": _selection(candidates, selected_id),
        "paired_identity_cluster_bootstrap": [
            _paired(selected_id, successor_id)
            for successor_id in candidate_ids
            if successor_id != selected_id
        ],
        "scope_interpretation": {
            "DEV": "MODEL_SELECTION_ONLY",
            "CAL": "CALIBRATION_REPORTING;NOT_SELECTION",
            "EXPOSED_DIAGNOSTIC": "RETROSPECTIVE_EXPOSED;NOT_FINAL_EVALUATION",
        },
        "contains_embeddings": False,
        "contains_sample_or_identity_tokens": False,
        "contains_ranked_qkv_traces": False,
        "limitations": [
            "DEV is used for successor selection; CAL and exposed diagnostics are not selection inputs.",
            "EXPOSED_DIAGNOSTIC is retrospective and is not an independent final evaluation.",
            "The report evaluates exact closed-set cosine retrieval only.",
        ],
    }
    return {**body, "public_report_sha256": content_sha256(body)}


def _governance() -> tuple[dict, dict, dict]:
    protocol_sha = _sha("protocol")
    protocol_bundle_sha = _sha("protocol-bundle")
    panel_sha = _sha("panel")
    panel_bundle_sha = _sha("panel-bundle")
    protocol = {
        "schema_version": "cvi.face_identity_protocol_bundle.v2",
        "protocol_sha256": protocol_sha,
        "bundle_sha256": protocol_bundle_sha,
        "protocol": {
            "score_bearing_bytes_used_for_role_allocation": False,
        },
        "census": {
            "identity_role_counts": {
                "FIT": 4,
                "DEV": 3,
                "CAL": 3,
                "EXPOSED_DIAGNOSTIC": 3,
                "EXCLUDED_UNSAFE_COMPONENT": 1,
            }
        },
    }
    panel = {
        "schema_version": "cvi.face_gallery_query_panel_bundle.v1",
        "panel_sha256": panel_sha,
        "bundle_sha256": panel_bundle_sha,
        "panel": {
            "source_protocol_sha256": protocol_sha,
            "source_protocol_bundle_sha256": protocol_bundle_sha,
            "score_inputs_used": False,
        },
        "census": {"common_k5_feasible_identity_count": 3},
    }
    successor_records = [
        {
            "sample_token": _sha(f"successor:{index}"),
            "dataset_name": "dogfacenet224" if index < 9 else "mpdd",
            "state": "USABLE",
            "artifact": {"full_mask_sha256": _sha(f"mask:successor:{index}")},
        }
        for index in range(12)
    ]
    auxiliary_records = [
        {
            "sample_token": _sha(f"auxiliary:{index}"),
            "dataset_name": "dogflw",
            "state": "USABLE" if index == 0 else "TERMINAL_EXCLUSION",
            "artifact": (
                {"full_mask_sha256": _sha("mask:auxiliary:0")} if index == 0 else None
            ),
        }
        for index in range(2)
    ]
    terminal_records = [
        {
            "sample_token": _sha("terminal:0"),
            "dataset_name": "mpdd",
            "state": "TERMINAL_EXCLUSION",
            "artifact": None,
        }
    ]
    inventory = {
        "schema_version": "cvi.full128_face_visible_successor_inventory_bundle.v1",
        "bundle_sha256": _sha("inventory-bundle"),
        "source_binding": {
            "face_protocol_v2_sha256": protocol_sha,
            "face_protocol_v2_bundle_sha256": protocol_bundle_sha,
            "gallery_query_panel_sha256": panel_sha,
            "gallery_query_panel_bundle_sha256": panel_bundle_sha,
        },
        "inventory": {
            "coverage": {
                "route_plan_sample_count": 15,
                "successor_sample_count": 12,
                "identity_free_auxiliary_sample_count": 2,
                "terminal_exclusion_count": 1,
                "state_counts": {"TERMINAL_EXCLUSION": 2, "USABLE": 13},
            },
            "crop_policy": {
                "source": "EXISTING_FULL128_MATERIALIZATION_ONLY",
                "recrop_permitted": False,
                "rgb_filename": "full.png",
                "mask_filename": "full-mask.png",
            },
            "successor_population": successor_records,
            "identity_free_auxiliary_population": auxiliary_records,
            "terminal_exclusions": terminal_records,
        },
    }
    return protocol, panel, inventory


def _patch_governance_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.full128_successors as successors
    import identity_governance.face_gallery_query_panel as gallery_panel
    import identity_governance.face_identity_protocol_v2 as protocol_v2
    from identity_methods.full_segment import face_visible

    monkeypatch.setattr(
        protocol_v2, "validate_face_identity_protocol_v2_bundle", lambda value: value
    )
    monkeypatch.setattr(
        gallery_panel, "validate_face_gallery_query_panel_bundle", lambda value: value
    )
    monkeypatch.setattr(
        face_visible,
        "validate_face_visible_successor_inventory_bundle",
        lambda value, *, verify_artifacts: value,
    )
    monkeypatch.setattr(
        successors,
        "build_authoritative_fixed_evaluation_panel",
        lambda inventory, protocol, panel: {"panel_sha256": _sha("evaluation-panel")},
    )


def test_public_successor_family_is_ordered_aggregate_only_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_governance_validators(monkeypatch)
    protocol, panel, inventory = _governance()
    figures = adapt_successor_family(
        _public_report(),
        protocol,
        panel,
        inventory,
        target_scope=PublicationScope.PUBLIC,
    )
    repeated = adapt_successor_family(
        _public_report(),
        protocol,
        panel,
        inventory,
        target_scope=PublicationScope.PUBLIC,
    )
    assert [figure.to_dict() for figure in repeated] == [
        figure.to_dict() for figure in figures
    ]
    assert [figure.figure_id for figure in figures] == [
        "00_evidence_ladder",
        "01_source_provenance",
        "02_census_availability",
        "03_role_dependency_closure",
        "04_governance_panel",
        "05_score_distributions",
        "06_model_dataflow",
        "07_evaluation_protocol",
        "08_cache_bindings",
        "09_embedding_spectrum_pca",
        "10_gallery_composition",
        "11_cosine_rank_distributions",
        "12_private_ranked_qkv",
        "13_primary_results_paired_deltas",
        "14_scope_interpretation",
        "15_limitations",
        "16_reproducibility_ledger",
        "17_evidence_release_ledger",
    ]
    serialized = json.dumps([figure.to_dict() for figure in figures], sort_keys=True)
    assert "S0" not in serialized and "S1" not in serialized
    assert "sample_token" not in serialized
    rank = next(figure for figure in figures if figure.figure_id.startswith("11_"))
    assert rank.payload["cosine_distribution"] is None
    assert "no cosine-score histogram" in rank.limitations[0]
    embedding = next(figure for figure in figures if figure.figure_id.startswith("09_"))
    assert embedding.payload["steps"][0]["detail"] == "UNAVAILABLE"
    private = next(figure for figure in figures if figure.figure_id.startswith("12_"))
    assert private.payload["steps"][0]["detail"] == "PRIVATE_NOT_PUBLISHED"
    runtime = next(figure for figure in figures if figure.figure_id.startswith("16_"))
    assert runtime.payload["steps"][0]["detail"] == "ON_DEVICE_NOT_ASSESSED_NO_TARGET"
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "artifact_contracts/schemas/cvi.figure_data.bundle.v1.schema.json"
    )
    schema_document = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema_document)
    for figure in figures:
        validator.validate(figure.to_bundle())


def test_successor_public_report_tamper_and_private_scope_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_governance_validators(monkeypatch)
    protocol, panel, inventory = _governance()
    report = _public_report()
    report["candidates"][0]["scope_aggregates"][0]["metrics"]["Rank-1"] = 0.0
    with pytest.raises(FigureContractError, match="public successor report contract"):
        adapt_successor_family(
            report,
            protocol,
            panel,
            inventory,
            target_scope=PublicationScope.PUBLIC,
        )
    with pytest.raises(PermissionError, match="private successor reports"):
        adapt_successor_family(
            _public_report(),
            protocol,
            panel,
            inventory,
            target_scope=PublicationScope.PUBLIC,
            private_report={"private": True},
            asset_root=Path("/tmp"),
        )


@pytest.mark.parametrize("attack", ("winner", "objective"))
def test_rehashed_wrong_dev_selection_is_rejected(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_governance_validators(monkeypatch)
    protocol, panel, inventory = _governance()
    report = _public_report()
    receipt = report["dev_selection_receipt"]
    receipt["selected_successor_id"] = "S0"
    if attack == "objective":
        receipt["candidates"][0]["objective_value"] = 0.99
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = content_sha256(receipt_payload)
    report_payload = {
        key: value for key, value in report.items() if key != "public_report_sha256"
    }
    report["public_report_sha256"] = content_sha256(report_payload)

    with pytest.raises(FigureContractError, match="public DEV selection"):
        adapt_successor_family(
            report,
            protocol,
            panel,
            inventory,
            target_scope=PublicationScope.PUBLIC,
        )


def test_actual_family_aliases_model_ladder_and_split_result_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_governance_validators(monkeypatch)
    protocol, panel, inventory = _governance()
    figures = adapt_successor_family(
        _public_report(
            candidate_ids=_ACTUAL_SUCCESSOR_IDS,
            selected_id="B5-SPATIAL",
        ),
        protocol,
        panel,
        inventory,
        target_scope=PublicationScope.PUBLIC,
    )
    ladder = next(figure for figure in figures if figure.figure_id.startswith("00_"))
    statuses = {item["alias"]: item["status"] for item in ladder.payload["variants"]}
    assert statuses["B5-Spatial"] == "GO"
    assert {alias for alias, status in statuses.items() if status == "GO"} == {
        "B5-Spatial"
    }
    assert [item["label"] for item in ladder.payload["boundaries"]] == [
        "DEV",
        "CAL",
        "EXPOSED",
    ]

    results = next(figure for figure in figures if figure.figure_id.startswith("13_"))
    assert {item["alias"] for item in results.payload["alias_legend"]} == {
        "B0-FV",
        "B1",
        "B2",
        "B3",
        "B4-U0",
        "B4-U1",
        "B5-Uniform",
        "B5-Channel",
        "B5-Spatial",
    }
    assert {row["scope"] for row in results.payload["absolute_rows"]} == {
        "DEV",
        "CAL",
        "EXPOSED_DIAGNOSTIC",
    }
    assert results.payload["delta_rows"]
    assert {row["metric"] for row in results.payload["delta_rows"]} == {
        "Rank-1",
        "MRR",
    }
    assert all(
        row["left_alias"] == "B5-Spatial" for row in results.payload["delta_rows"]
    )

    chapters = {figure.figure_id[:2]: figure for figure in figures}
    assert sum(row["count"] for row in chapters["01"].payload["rows"]) == 15
    assert {
        node["label"].split("\n", 1)[0] for node in chapters["04"].payload["nodes"]
    } >= {"Existing Full128 materialization", "RGB + mask bindings"}
    assert any(
        step["detail"].endswith("UNAVAILABLE_NO_PUBLIC_QUALITATIVE_ARTIFACT")
        for step in chapters["05"].payload["steps"]
    )
    assert any(
        "Frozen DINOv2 patches" in node["label"]
        for node in chapters["07"].payload["nodes"]
    )
    assert any(
        "UNAVAILABLE_NO_TRAINING_ARTIFACT_SUPPLIED" in step["detail"]
        for step in chapters["08"].payload["steps"]
    )
    control_steps = chapters["14"].payload["steps"]
    assert {step["label"].split(" - ")[-1] for step in control_steps} == {
        "B4-U0",
        "B4-U1",
        "B5-Uniform",
        "B5-Channel",
    }
    assert all("Rank-1 +0.1000" in step["detail"] for step in control_steps)
    robustness = chapters["15"].payload["steps"]
    assert any("B5-Uniform" == step["label"] for step in robustness)
    assert any(
        "UNASSESSED_NO_MASK_PERTURBATION_ARTIFACT" in step["detail"]
        for step in robustness
    )
    runtime = chapters["16"].payload["steps"]
    assert runtime[0]["detail"] == "ON_DEVICE_NOT_ASSESSED_NO_TARGET"
    assert any("UNASSESSED_NO_LATENCY_ARTIFACT" in step["detail"] for step in runtime)


def test_successor_public_bundle_renders_in_chapter_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    _patch_governance_validators(monkeypatch)
    protocol, panel, inventory = _governance()
    descriptors = tuple(
        {
            "schema_version": "cvi.full128_successor_embedding_cache.v1",
            "successor_id": successor_id,
            "cache_descriptor_sha256": _sha(f"descriptor:{successor_id}"),
            "sample_tokens": [_sha(f"sample:{index}") for index in range(12)],
            "pack_path": f"/private/cache/{successor_id}.f32le",
        }
        for successor_id in _ACTUAL_SUCCESSOR_IDS
    )
    matrices = {}
    for index, successor_id in enumerate(_ACTUAL_SUCCESSOR_IDS):
        rng = np.random.default_rng(index)
        matrix = rng.normal(size=(12, 128)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        matrices[successor_id] = matrix
    import evaluation.full128_successors as successors

    monkeypatch.setattr(
        successors,
        "open_successor_embedding_cache",
        lambda descriptor, *, successor_inventory_bundle, evaluation_panel: (
            SimpleNamespace(
                descriptor=descriptor,
                load_embeddings=lambda tokens, successor_id=descriptor["successor_id"]: (
                    matrices[successor_id].copy()
                ),
            )
        ),
    )
    figures = adapt_successor_family(
        _public_report(
            candidate_ids=_ACTUAL_SUCCESSOR_IDS,
            selected_id="B5-SPATIAL",
            cache_hashes={
                descriptor["successor_id"]: descriptor["cache_descriptor_sha256"]
                for descriptor in descriptors
            },
        ),
        protocol,
        panel,
        inventory,
        target_scope=PublicationScope.PUBLIC,
        cache_descriptors=descriptors,
    )
    assert len(figures) == 18
    diagnostics = next(
        figure for figure in figures if figure.figure_id.startswith("09_")
    )
    assert len(diagnostics.payload["manifest"]) == 9
    assert {
        row["alias"] for row in diagnostics.payload["manifest"] if row["displayed"]
    } == {
        "B0-FV",
        "B2",
        "B3",
        "B4-U1",
        "B5-Spatial",
    }
    assert {row["label"] for row in diagnostics.payload["series"]} == {
        "B0-FV",
        "B2",
        "B3",
        "B4-U1",
        "B5-Spatial",
    }
    output = tmp_path / "successor-publication"
    publish(figures, output, target_scope=PublicationScope.PUBLIC)
    output_inventory = json.loads(
        (output / "output_inventory.json").read_text(encoding="utf-8")
    )
    svg_chapters = [
        entry["path"].split("/", 1)[1].removesuffix(".svg")
        for entry in output_inventory["entries"]
        if entry["path"].endswith(".svg")
    ]
    assert svg_chapters == [figure.figure_id for figure in figures]
    for figure in figures:
        with Image.open(output / "figures" / f"{figure.figure_id}.png") as rendered:
            assert rendered.size == (1280, 720)
    serialized = b"".join(
        path.read_bytes()
        for path in output.rglob("*")
        if path.suffix in {".svg", ".html", ".json"}
    )
    assert b"S0" not in serialized and b"S1" not in serialized


def test_cache_descriptors_produce_aggregate_pca_without_tokens_or_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_governance_validators(monkeypatch)
    protocol, panel, inventory = _governance()
    descriptor = {
        "schema_version": "cvi.full128_successor_embedding_cache.v1",
        "successor_id": "S1",
        "cache_descriptor_sha256": _sha("descriptor:S1"),
        "sample_tokens": [_sha(f"sample:{index}") for index in range(4)],
        "pack_path": "/private/cache/S1.f32le",
    }
    report = _public_report(
        cache_hashes={
            "S0": _sha("cache:S0"),
            "S1": descriptor["cache_descriptor_sha256"],
        }
    )
    matrix = np.zeros((4, 128), dtype=np.float32)
    matrix[np.arange(4), np.arange(4)] = 1.0
    fake_cache = SimpleNamespace(
        descriptor=descriptor,
        load_embeddings=lambda tokens: matrix.copy(),
    )
    import evaluation.full128_successors as successors

    monkeypatch.setattr(
        successors,
        "build_authoritative_fixed_evaluation_panel",
        lambda inventory, protocol, panel: {"panel_sha256": _sha("evaluation-panel")},
    )
    monkeypatch.setattr(
        successors,
        "open_successor_embedding_cache",
        lambda descriptor, *, successor_inventory_bundle, evaluation_panel: fake_cache,
    )
    figures = adapt_successor_family(
        report,
        protocol,
        panel,
        inventory,
        target_scope=PublicationScope.PUBLIC,
        cache_descriptors=(descriptor,),
    )
    diagnostics = next(
        figure for figure in figures if figure.figure_id.startswith("09_")
    )
    serialized = json.dumps(diagnostics.to_dict(), sort_keys=True)
    assert "sample_tokens" not in serialized
    assert "/private/cache" not in serialized
    assert diagnostics.payload["series"][0]["sample_count"] == 4


def _private_and_public(tmp_path: Path) -> tuple[dict, dict, dict]:
    query_token = _sha("query")
    query_path = tmp_path / "query.png"
    Image.new("RGB", (40, 30), (90, 130, 160)).save(query_path)
    query_hash = hashlib.sha256(query_path.read_bytes()).hexdigest()
    key_records = []
    ranked_rows = []
    for index in range(8):
        rank = index + 1
        token = _sha(f"key:{rank}")
        path = tmp_path / f"key-{rank}.png"
        Image.new(
            "RGB",
            (40, 30),
            (100 + index * 12, 80 + index * 8, 150 - index * 9),
        ).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relevant = rank in {1, 4}
        key_records.append(
            {
                "sample_token": token,
                "registered_identity_id": (
                    "private-identity-a" if relevant else "private-identity-b"
                ),
                "dataset_name": "dogfacenet224",
                "state": "USABLE",
                "artifact": {
                    "full_rgb_path": str(path),
                    "full_rgb_sha256": digest,
                    "full_mask_path": str(path),
                    "full_mask_sha256": digest,
                },
            }
        )
        ranked_rows.append(
            {
                "rank": rank,
                "score": 0.92 - index * 0.07,
                "K": {
                    "winning_template_row": index,
                    "sample_token": token,
                    "embedding": [1.0] + [0.0] * 127,
                },
                "V": {
                    "registered_identity_id": (
                        "private-identity-a" if relevant else "private-identity-b"
                    ),
                    "template_id": f"private-template-{rank}",
                    "content_sha256": _sha(f"content:{rank}"),
                },
            }
        )
    candidates = []
    for successor_id, base in (("S0", 0.65), ("S1", 0.75)):
        aggregates = [
            _scope(successor_id, scope, base - index * 0.05)
            for index, scope in enumerate(("DEV", "CAL", "EXPOSED_DIAGNOSTIC"))
        ]
        trace = []
        if successor_id == "S1":
            trace = [
                {
                    "scope": "DEV",
                    "dataset_name": "dogfacenet224",
                    "enrollment_k": 1,
                    "Q": {
                        "sample_token": query_token,
                        "embedding": [1.0] + [0.0] * 127,
                    },
                    "ranked_KV": ranked_rows,
                    "exact_cosine": True,
                }
            ]
        private_gallery = [
            {**_gallery(scope, k), "reopened_read_only": True, "exact_cosine": True}
            for scope in ("DEV", "CAL", "EXPOSED_DIAGNOSTIC")
            for k in (1, 3, 5)
        ]
        body = {
            "successor_id": successor_id,
            "cache_descriptor_sha256": _sha(f"cache:{successor_id}"),
            "cohort_results": [],
            "scope_aggregates": aggregates,
            "gallery_bindings": private_gallery,
            "ranked_private_qkv_traces": trace,
        }
        candidates.append({**body, "candidate_report_sha256": content_sha256(body)})
    selection = _selection(candidates, "S1")
    private_body = {
        "schema_version": "cvi.full128_successor_private_evaluation.v1",
        "visibility": "PRIVATE",
        "successor_inventory_bundle_sha256": _sha("inventory-bundle"),
        "evaluation_panel": {"panel_sha256": _sha("evaluation-panel")},
        "candidates": candidates,
        "dev_selection_receipt": selection,
        "paired_identity_cluster_bootstrap": [_paired("S1", "S0")],
        "scope_interpretation": {
            "DEV": "MODEL_SELECTION_ONLY",
            "CAL": "CALIBRATION_REPORTING;NOT_SELECTION",
            "EXPOSED_DIAGNOSTIC": "RETROSPECTIVE_EXPOSED;NOT_FINAL_EVALUATION",
        },
    }
    private = {**private_body, "report_sha256": content_sha256(private_body)}
    public = sanitize_successor_evaluation_report(private)
    inventory_records = [
        {
            "sample_token": query_token,
            "registered_identity_id": "private-identity-a",
            "dataset_name": "dogfacenet224",
            "state": "USABLE",
            "artifact": {
                "full_rgb_path": str(query_path),
                "full_rgb_sha256": query_hash,
                "full_mask_path": str(query_path),
                "full_mask_sha256": query_hash,
            },
        },
        *key_records,
    ]
    return private, public, {"records": inventory_records}


def test_private_ranked_qkv_requires_asset_root_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    _patch_governance_validators(monkeypatch)
    protocol, panel, inventory = _governance()
    private, public, assets = _private_and_public(tmp_path)
    inventory["inventory"]["successor_population"] = assets["records"]
    inventory["inventory"]["coverage"]["successor_sample_count"] = 9
    inventory["inventory"]["coverage"]["route_plan_sample_count"] = 12
    with pytest.raises(ValueError, match="explicit asset root"):
        adapt_successor_family(
            public,
            protocol,
            panel,
            inventory,
            target_scope=PublicationScope.PRIVATE,
            private_report=private,
        )
    figures = adapt_successor_family(
        public,
        protocol,
        panel,
        inventory,
        target_scope=PublicationScope.PRIVATE,
        private_report=private,
        asset_root=tmp_path,
    )
    contact = next(figure for figure in figures if figure.figure_id.startswith("12_"))
    assert contact.scope is PublicationScope.PRIVATE
    assert contact.payload["query"]["path"] == "query.png"
    assert len(contact.payload["candidates"]) == 8
    assert all(
        item["margin"] == pytest.approx(0.07)
        for item in contact.payload["candidates"][:-1]
    )
    assert contact.payload["candidates"][-1]["margin"] is None
    assert {
        item["rank"]
        for item in contact.payload["candidates"]
        if item["outcome"] == "relevant"
    } == {1, 4}
    output = tmp_path / "publication"
    publish(
        figures,
        output,
        target_scope=PublicationScope.PRIVATE,
        asset_root=tmp_path,
        figure_ids=("12_private_ranked_qkv",),
    )
    assert (output / "figures" / "12_private_ranked_qkv.svg").is_file()
    with Image.open(output / "figures" / "12_private_ranked_qkv.png") as rendered:
        assert rendered.size == (1280, 720)


def test_successor_family_workflow_help_exposes_named_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "workflows/render_research_visualizations.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "successor-family",
        "--public-report",
        "--private-report",
        "--face-protocol-v2",
        "--gallery-query-panel",
        "--successor-inventory",
        "--cache-descriptor",
    ):
        assert option in result.stdout
