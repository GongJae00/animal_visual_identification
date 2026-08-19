"""Generate bounded private B3/B5-SPATIAL traces from production artifacts."""

from __future__ import annotations

from legacy.version.root import repository_root as find_repo_root
import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.full_segment.full128_analysis import build_executed_representation_trace_manifest
from evaluation.full_segment.full128_successors import (
    open_successor_embedding_cache,
    validate_fixed_evaluation_panel,
)
from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_directory_bundle,
)
from foundation.protected_publication import admit_new_external_output
from foundation.provenance import content_sha256
from embedding.methods.full_segment.preparation.data import read_full128_crop
from embedding.methods.full_segment.face_visible import (
    validate_face_visible_successor_inventory_bundle,
)
from embedding.methods.full_segment.models.successor_models import (
    load_receipt_bound_dinov2_patch_backbone,
)
from embedding.learning.full_segment.full128_successor_production import (
    _sample_from_row,
    prepare_production_runtime,
    restore_successor_trace_context,
    validate_production_config,
)

_CANDIDATES = ("B3", "B5-SPATIAL")
_LARGE_JSON = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-run", required=True, type=Path)
    parser.add_argument("--successor-inventory", required=True, type=Path)
    parser.add_argument("--evaluation-panel", required=True, type=Path)
    parser.add_argument("--dinov2-model-directory", required=True, type=Path)
    parser.add_argument("--dinov2-weight-intake-bundle", required=True, type=Path)
    parser.add_argument("--dinov2-preprocessor-intake-bundle", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _new_external_output(args.output_directory)
    run_manifest = read_strict_json_document(
        args.production_run / "run-manifest.json", maximum_bytes=16_777_216
    ).payload
    config = validate_production_config(run_manifest.get("config"))
    prepare_production_runtime(config)
    backbone, contract = load_receipt_bound_dinov2_patch_backbone(
        model_directory=args.dinov2_model_directory,
        weight_intake_bundle=args.dinov2_weight_intake_bundle,
        preprocessor_intake_bundle=args.dinov2_preprocessor_intake_bundle,
    )
    context = restore_successor_trace_context(
        run_directory=args.production_run,
        dinov2_backbone=backbone,
        dinov2_contract=contract,
    )
    inventory = validate_face_visible_successor_inventory_bundle(
        read_strict_json_document(args.successor_inventory, **_LARGE_JSON).payload,
        verify_artifacts=False,
    )
    panel = validate_fixed_evaluation_panel(
        read_strict_json_document(args.evaluation_panel, **_LARGE_JSON).payload
    )
    if (
        inventory["bundle_sha256"]
        != context["run_manifest"]["successor_inventory_bundle_sha256"]
        or inventory["inventory_sha256"]
        != context["run_manifest"]["successor_inventory_sha256"]
        or panel["panel_sha256"] != context["run_manifest"]["evaluation_panel_sha256"]
    ):
        raise ValueError("trace inventory or evaluation panel differs from production")

    caches = {
        candidate_id: open_successor_embedding_cache(
            context["cache_descriptors"][candidate_id],
            successor_inventory_bundle=inventory,
            evaluation_panel=panel,
        )
        for candidate_id in _CANDIDATES
    }
    query_token, gallery_tokens = _bounded_pair_population(panel)
    key_tokens: dict[str, str] = {}
    cached_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate_id, cache in caches.items():
        query_embedding = cache.load_embeddings((query_token,))[0]
        gallery_embeddings = cache.load_embeddings(gallery_tokens)
        scores = gallery_embeddings @ query_embedding
        winner = int(np.argmax(scores))
        key_tokens[candidate_id] = gallery_tokens[winner]
        cached_pairs[candidate_id] = (query_embedding, gallery_embeddings[winner])

    selected_tokens = tuple(sorted({query_token, *key_tokens.values()}))
    rows = {
        row["sample_token"]: row
        for row in inventory["inventory"]["successor_population"]
    }
    if missing := set(selected_tokens) - set(rows):
        raise ValueError(f"trace samples are absent from inventory: {len(missing)}")
    samples = {token: _sample_from_row(rows[token]) for token in selected_tokens}
    live_tokens, live_occupancy, live_indices = _execute_live_dinov2(
        context, selected_tokens, rows
    )
    token_cache = context["token_cache"]
    token_cache_index = {
        token: index for index, token in enumerate(token_cache["sample_tokens"])
    }
    device = torch.device(config["precision"]["device"])
    traces: list[dict[str, Any]] = []
    for candidate_id in _CANDIDATES:
        key_token = key_tokens[candidate_id]
        model = context["models"][candidate_id].to(device).eval()
        candidate_run = context["candidate_runs"][candidate_id]
        descriptor = context["cache_descriptors"][candidate_id]
        checkpoint = read_strict_json_document(
            context["root"] / candidate_id / "checkpoint" / "checkpoint-manifest.json",
            maximum_bytes=16_777_216,
        ).payload
        cached_query, cached_key = cached_pairs[candidate_id]
        trace = build_executed_representation_trace_manifest(
            successor_id=candidate_id,
            model=model,
            query_sample_token=query_token,
            key_sample_token=key_token,
            cached_query_tokens=_cached_tensor(
                token_cache["tokens"],
                token_cache_index[query_token],
                config["extraction_batch_size"],
                device,
            ),
            cached_key_tokens=_cached_tensor(
                token_cache["tokens"],
                token_cache_index[key_token],
                config["extraction_batch_size"],
                device,
            ),
            cached_query_occupancy=_cached_tensor(
                token_cache["occupancy"],
                token_cache_index[query_token],
                config["extraction_batch_size"],
                device,
            ),
            cached_key_occupancy=_cached_tensor(
                token_cache["occupancy"],
                token_cache_index[key_token],
                config["extraction_batch_size"],
                device,
            ),
            live_query_tokens=live_tokens[query_token],
            live_key_tokens=live_tokens[key_token],
            live_query_occupancy=live_occupancy[query_token],
            live_key_occupancy=live_occupancy[key_token],
            cached_query_embedding=cached_query,
            cached_key_embedding=cached_key,
            model_input_transform={
                "source_size": [224, 224],
                "model_input_size": [224, 224],
                "color_mode": "RGB",
                "resize_interpolation": "NONE_ALREADY_224X224",
                "mask_application": "IMAGENET_MEAN_NEUTRAL_BEFORE_NORMALIZATION",
                "channel_mean": [0.485, 0.456, 0.406],
                "channel_std": [0.229, 0.224, 0.225],
            },
            artifact_bindings=_artifact_bindings(
                context, candidate_run, descriptor, checkpoint
            ),
            query_input_binding=_sample_binding(samples[query_token]),
            key_input_binding=_sample_binding(samples[key_token]),
            rank=1,
            query_index=live_indices[query_token],
            key_index=live_indices[key_token],
        )
        traces.append(trace)

    generation_payload = {
        "schema_version": "cvi.full128_representation_trace_generation.v1",
        "visibility": "PRIVATE",
        "selection_policy": "FIRST_SORTED_AVAILABLE_COHORT_AND_QUERY;EXACT_TOP1_TEMPLATE",
        "trace_count": len(traces),
        "trace_sha256s": [trace["trace_sha256"] for trace in traces],
        "run_manifest_sha256": context["run_manifest"]["run_manifest_sha256"],
        "evaluation_panel_sha256": panel["panel_sha256"],
    }
    generation = {
        **generation_payload,
        "generation_sha256": content_sha256(generation_payload),
    }
    strategy = write_private_json_directory_bundle(
        output,
        (
            ("B3-private-trace.json", traces[0]),
            ("B5-SPATIAL-private-trace.json", traces[1]),
            ("generation-manifest.json", generation),
        ),
    )
    print(
        json.dumps(
            {
                "status": "GENERATED_ACTUAL_FULL128_REPRESENTATION_TRACES",
                "output_directory": os.fspath(output),
                "trace_sha256s": generation["trace_sha256s"],
                "generation_sha256": generation["generation_sha256"],
                "publication_strategy": strategy,
            },
            sort_keys=True,
        )
    )
    return 0


def _bounded_pair_population(panel: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    cohorts = sorted(
        (row for row in panel["cohorts"] if row["status"] == "AVAILABLE"),
        key=lambda row: (row["scope"], row["dataset_name"], row["enrollment_k"]),
    )
    if not cohorts:
        raise ValueError("trace evaluation panel has no available cohort")
    cohort = cohorts[0]
    queries = tuple(cohort["query_sample_tokens"])
    gallery = tuple(cohort["gallery_sample_tokens"])
    if not queries or not gallery:
        raise ValueError("trace cohort lacks query or gallery samples")
    return queries[0], gallery


def _execute_live_dinov2(
    context: Mapping[str, Any],
    sample_tokens: Sequence[str],
    inventory_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    device = torch.device(context["config"]["precision"]["device"])
    model = context["models"]["B3"].to(device).eval()
    population = context["token_cache"]["sample_tokens"]
    population_index = {token: index for index, token in enumerate(population)}
    batch_size = context["config"]["extraction_batch_size"]
    batch_starts = sorted(
        {population_index[token] // batch_size * batch_size for token in sample_tokens}
    )
    live_tokens: dict[str, torch.Tensor] = {}
    live_occupancy: dict[str, torch.Tensor] = {}
    live_indices: dict[str, int] = {}
    selected = set(sample_tokens)
    for start in batch_starts:
        batch_tokens = population[start : start + batch_size]
        rgbs = []
        masks = []
        for token in batch_tokens:
            sample = _sample_from_row(inventory_rows[token])
            rgb, mask = read_full128_crop(sample)
            rgbs.append(torch.from_numpy(rgb.transpose(2, 0, 1).copy()))
            masks.append(torch.from_numpy(mask[None, ...].copy()))
        rgb_tensor = (
            torch.stack(rgbs).to(device=device, dtype=torch.float32).div_(255.0)
        )
        mask_tensor = torch.stack(masks).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            tokens, occupancy = model.extract_tokens(rgb_tensor, mask_tensor)
        for offset, token in enumerate(batch_tokens):
            if token in selected:
                live_tokens[token] = tokens
                live_occupancy[token] = occupancy
                live_indices[token] = offset
    if set(live_tokens) != selected or set(live_occupancy) != selected:
        raise RuntimeError("live DINOv2 trace extraction coverage differs")
    return live_tokens, live_occupancy, live_indices


def _cached_tensor(
    values: np.ndarray, index: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    start = index // batch_size * batch_size
    return torch.from_numpy(np.asarray(values[start : start + batch_size]).copy()).to(
        device
    )


def _artifact_bindings(
    context: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, str]:
    token_manifest = context["token_cache"]["manifest"]
    dino = context["run_manifest"]["dinov2"]
    return {
        "run_manifest_sha256": context["run_manifest"]["run_manifest_sha256"],
        "candidate_run_sha256": candidate_run["candidate_run_sha256"],
        "model_manifest_sha256": candidate_run["model_manifest_sha256"],
        "checkpoint_manifest_sha256": candidate_run["checkpoint_manifest_sha256"],
        "checkpoint_state_sha256": checkpoint["state"]["sha256"],
        "preprocessing_manifest_sha256": candidate_run["preprocessing_manifest_sha256"],
        "embedding_manifest_sha256": candidate_run["embedding_manifest_sha256"],
        "token_cache_manifest_sha256": token_manifest["cache_manifest_sha256"],
        "token_cache_tokens_sha256": token_manifest["tokens"]["sha256"],
        "token_cache_occupancy_sha256": token_manifest["occupancy"]["sha256"],
        "evaluation_cache_descriptor_sha256": descriptor["cache_descriptor_sha256"],
        "evaluation_pack_sha256": descriptor["pack_sha256"],
        "dinov2_model_sha256": dino["model_sha256"],
        "dinov2_config_sha256": dino["config_sha256"],
        "dinov2_preprocessor_sha256": dino["preprocessor_sha256"],
    }


def _sample_binding(sample: Any) -> dict[str, str]:
    return {
        "rgb_sha256": sample.rgb_sha256,
        "mask_sha256": sample.mask_sha256,
        "crop_record_sha256": sample.crop_record_sha256,
    }


def _new_external_output(path: Path) -> Path:
    return admit_new_external_output(
        path,
        repository_root=find_repo_root(__file__),
        repository_error="representation trace output must remain outside repository",
        overwrite_error="refusing to overwrite representation trace output",
    )


if __name__ == "__main__":
    raise SystemExit(main())
