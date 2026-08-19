"""Render private Full128 audit evidence as nine atomic PNG plates only."""

from __future__ import annotations

from legacy.version.root import repository_root as find_repo_root
import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.full_segment.full128_analysis import validate_executed_representation_trace_manifest
from evaluation.full_segment.full128_successors import (
    build_authoritative_fixed_evaluation_panel,
    open_successor_embedding_cache,
    sanitize_successor_evaluation_report,
    validate_fixed_evaluation_panel,
)
from foundation.protected_io import read_strict_json_document
from embedding.methods.full_segment.preparation.data import (
    Full128Sample,
    read_full128_crop,
    read_full128_mask,
)
from embedding.methods.full_segment.face_visible import (
    validate_face_visible_successor_inventory_bundle,
)
from parsing.full_segment.full_segment_cache import validate_full_segment_cache_bundle
from visualization.full128_visual_audit import (
    AuditSample,
    QueryOutcome,
    reconstruct_ranking,
    relevant_rank,
    render_png_audit,
    select_outcome_strata,
)

_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    # Final private reports retain ranked 128D traces and exceed the ordinary
    # metadata-only report node budget.
    "maximum_nodes": 100_000_000,
    "maximum_keys": 20_000_000,
    "maximum_array_length": 1_000_000,
}
_LANES = {
    "successor": ("dogfacenet224", "mpdd"),
    "auxiliary": ("ap10k-dog", "dogflw", "oxford-pets-dog"),
    "terminal": ("dogfacenet224", "sibetan", "yt-bb-dog"),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _prepare_output(args.output_dir)
    inventory = validate_face_visible_successor_inventory_bundle(
        _read(args.successor_inventory), verify_artifacts=False
    )
    panel = _load_panel(args, inventory)
    asset_root = args.asset_root.resolve(strict=True)
    if asset_root != Path(inventory["inventory"]["artifact_root"]).resolve(strict=True):
        raise ValueError("asset root differs from successor inventory binding")
    b3_descriptor, b5_descriptor = (
        _read(args.b3_cache_descriptor),
        _read(args.b5_cache_descriptor),
    )
    b3 = open_successor_embedding_cache(
        b3_descriptor, successor_inventory_bundle=inventory, evaluation_panel=panel
    )
    b5 = open_successor_embedding_cache(
        b5_descriptor, successor_inventory_bundle=inventory, evaluation_panel=panel
    )
    try:
        if (
            b3.descriptor["successor_id"] != "B3"
            or b5.descriptor["successor_id"] != "B5-SPATIAL"
        ):
            raise ValueError("audit requires B3 and B5-SPATIAL cache descriptors")
        report = _read(args.private_report)
        sanitize_successor_evaluation_report(report)
        if (
            validate_fixed_evaluation_panel(report["evaluation_panel"])["panel_sha256"]
            != panel["panel_sha256"]
        ):
            raise ValueError("private report evaluation panel differs")
        trace = validate_executed_representation_trace_manifest(_read(args.b5_trace))
        _validate_trace_binding(trace, inventory, b5.descriptor)
        records = _records(inventory, asset_root)
        b3_vectors = _vectors(b3)
        b5_vectors = _vectors(b5)
        report_ranks = _report_ranks(report, b3.descriptor, b5.descriptor)
        outcomes, dev_population = _dev_outcomes(
            panel, records, b3_vectors, b5_vectors, report_ranks
        )
        strata = select_outcome_strata(outcomes)
        gallery_query = min(outcomes, key=lambda item: (item.cohort_key, item.token))
        samples = _load_samples(
            records,
            {
                gallery_query.token,
                *(item.token for item in strata.values()),
                *(
                    ranked.token
                    for item in (gallery_query, *strata.values())
                    for ranked in (*item.b3_ranked[:5], *item.b5_ranked[:5])
                ),
                trace["private_samples"]["query_sample_token"],
                trace["private_samples"]["key_sample_token"],
            },
        )
        input_lanes = _input_lanes(records)
        for lane in input_lanes.values():
            for _, selected in lane:
                for sample in selected:
                    samples.setdefault(sample.token, sample)
        render_png_audit(
            output_dir=output,
            input_lanes=input_lanes,
            gallery_query=gallery_query,
            outcomes=strata,
            samples=samples,
            dev_population=dev_population,
            b3_vectors=b3_vectors,
            b5_vectors=b5_vectors,
            trace=trace,
        )
    finally:
        b3.close()
        b5.close()
    print(
        json.dumps(
            {
                "status": "RENDERED_PRIVATE_FULL128_VISUAL_AUDIT",
                "output_directory": str(output),
                "rendered_filenames": sorted(path.name for path in output.iterdir()),
                "selected_strata": {
                    name: {
                        "relevant_rank": item.relevant_rank,
                        "b5_top1_margin": item.margin,
                        "basis": _selection_basis(name),
                    }
                    for name, item in strata.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-inventory", required=True, type=Path)
    parser.add_argument("--evaluation-panel", type=Path)
    parser.add_argument("--face-protocol-v2", type=Path)
    parser.add_argument("--gallery-query-panel", type=Path)
    parser.add_argument("--b3-cache-descriptor", required=True, type=Path)
    parser.add_argument("--b5-cache-descriptor", required=True, type=Path)
    parser.add_argument("--private-report", required=True, type=Path)
    parser.add_argument("--b5-trace", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _read(path: Path) -> dict[str, Any]:
    return read_strict_json_document(path, **_LIMITS).payload


def _load_panel(
    args: argparse.Namespace, inventory: Mapping[str, Any]
) -> dict[str, Any]:
    sources = (args.face_protocol_v2, args.gallery_query_panel)
    if args.evaluation_panel is not None:
        if any(sources):
            raise ValueError(
                "provide an effective evaluation panel or both governance inputs"
            )
        return validate_fixed_evaluation_panel(_read(args.evaluation_panel))
    if any(value is None for value in sources):
        raise ValueError("both face protocol v2 and gallery/query panel are required")
    return build_authoritative_fixed_evaluation_panel(
        inventory, _read(args.face_protocol_v2), _read(args.gallery_query_panel)
    )


def _prepare_output(path: Path) -> Path:
    repository = find_repo_root(__file__)
    requested = path.absolute()
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    if output.is_symlink():
        raise ValueError("audit output must not be a symlink")
    if output.is_relative_to(repository) and output != repository / "Visualization":
        raise ValueError(
            "repository audit output is restricted to ignored Visualization"
        )
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("audit output must be an empty directory")
    else:
        output.mkdir(mode=0o700)
    return output


def _records(
    inventory: Mapping[str, Any], asset_root: Path
) -> dict[str, dict[str, Any]]:
    all_records = [
        *inventory["inventory"]["successor_population"],
        *inventory["inventory"]["identity_free_auxiliary_population"],
        *inventory["inventory"].get("terminal_exclusions", ()),
    ]
    records = {row["sample_token"]: row for row in all_records}
    if len(records) != len(all_records):
        raise ValueError("inventory repeats sample tokens")
    for row in records.values():
        artifact = row.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        for key in ("full_rgb_path", "full_mask_path"):
            if artifact[key] is not None and not Path(artifact[key]).resolve(
                strict=False
            ).is_relative_to(asset_root):
                raise ValueError(
                    "inventory artifact is outside the declared asset root"
                )
    return records


def _sample(row: Mapping[str, Any]) -> Full128Sample:
    artifact = row.get("artifact")
    if not isinstance(artifact, Mapping):
        raise TypeError("audit sample artifact binding differs")
    if any(
        artifact[key] is None
        for key in (
            "full_rgb_path",
            "full_rgb_sha256",
            "full_mask_path",
            "full_mask_sha256",
            "crop_record_sha256",
        )
    ):
        raise ValueError("audit sample lacks bound crop artifacts")
    return Full128Sample(
        sample_id=row["sample_token"],
        identity_id=row["registered_identity_id"] or "",
        dataset_name=row["dataset_name"],
        view="full",
        role="AUDIT",
        rgb_path=Path(artifact["full_rgb_path"]),
        rgb_sha256=artifact["full_rgb_sha256"],
        mask_path=Path(artifact["full_mask_path"]),
        mask_sha256=artifact["full_mask_sha256"],
        crop_record_sha256=artifact["crop_record_sha256"],
    )


def _route(row: Mapping[str, Any]) -> str:
    artifact = row.get("artifact")
    if not isinstance(artifact, Mapping):
        raise TypeError("audit sample artifact binding differs")
    cache_path = Path(artifact["full_rgb_path"]).parent / "full-segment-cache.json"
    cache_bundle = _read(cache_path)
    cache = validate_full_segment_cache_bundle(cache_bundle)
    if cache_bundle["cache_sha256"] != artifact["full_segment_cache_sha256"]:
        raise ValueError("audit crop cache binding differs")
    records = cache["records"]
    if len(records) != 1 or not isinstance(records[0]["crop"], Mapping):
        raise ValueError("audit crop cache record differs")
    crop = records[0]["crop"]
    if (
        any(
            crop[f"full_{key}_sha256"] != artifact[f"full_{key}_sha256"]
            for key in ("rgb", "mask")
        )
        or crop["crop_record_sha256"] != artifact["crop_record_sha256"]
    ):
        raise ValueError("audit crop cache artifact binding differs")
    route = crop["route"]
    if not isinstance(route, str):
        raise TypeError("audit crop route differs")
    return route


def _load_samples(
    records: Mapping[str, Mapping[str, Any]], tokens: set[str]
) -> dict[str, AuditSample]:
    loaded: dict[str, AuditSample] = {}
    for token in sorted(tokens):
        row = records[token]
        rgb, mask = read_full128_crop(_sample(row))
        loaded[token] = AuditSample(
            token,
            row["registered_identity_id"],
            row["dataset_name"],
            rgb,
            mask,
            _route(row),
        )
    return loaded


def _input_lanes(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[tuple[str, tuple[AuditSample, ...]]]]:
    result: dict[str, list[tuple[str, tuple[AuditSample, ...]]]] = {}
    selected_by_dataset: dict[str, tuple[AuditSample, ...]] = {}
    for lane, datasets in _LANES.items():
        rendered: list[tuple[str, tuple[AuditSample, ...]]] = []
        for dataset in datasets:
            if dataset not in selected_by_dataset:
                selected_by_dataset[dataset] = tuple(
                    AuditSample(
                        row["sample_token"],
                        row["registered_identity_id"],
                        dataset,
                        *read_full128_crop(_sample(row)),
                        _route(row),
                    )
                    for row in _select_occupancy_rows(records, dataset)
                )
            rendered.append((dataset, selected_by_dataset[dataset]))
        result[lane] = rendered
    return result


def _select_occupancy_rows(
    records: Mapping[str, Mapping[str, Any]], dataset: str
) -> tuple[Mapping[str, Any], ...]:
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for row in records.values():
        if row["dataset_name"] != dataset:
            continue
        artifact = row.get("artifact")
        if not isinstance(artifact, Mapping) or any(
            artifact[key] is None for key in ("full_rgb_path", "full_mask_path")
        ):
            continue
        candidates.append(
            (
                float(read_full128_mask(_sample(row)).mean()),
                row["sample_token"],
                row,
            )
        )
    if not candidates:
        return ()
    candidates.sort(key=lambda item: (item[0], item[1]))
    indices = (0, (len(candidates) - 1) // 2, len(candidates) - 1)
    return tuple(candidates[index][2] for index in indices)


def _vectors(cache: Any) -> dict[str, Any]:
    tokens = cache.descriptor["sample_tokens"]
    return dict(zip(tokens, cache.load_embeddings(tokens), strict=True))


def _report_ranks(
    report: Mapping[str, Any],
    b3_descriptor: Mapping[str, Any],
    b5_descriptor: Mapping[str, Any],
) -> dict[str, dict[tuple[str, str, int, str], int]]:
    result: dict[str, dict[tuple[str, str, int, str], int]] = {}
    expected = {
        "B3": b3_descriptor["cache_descriptor_sha256"],
        "B5-SPATIAL": b5_descriptor["cache_descriptor_sha256"],
    }
    for candidate in report["candidates"]:
        identifier = candidate["successor_id"]
        if identifier not in expected:
            continue
        if candidate["cache_descriptor_sha256"] != expected[identifier]:
            raise ValueError(f"private report {identifier} cache descriptor differs")
        result[identifier] = {
            (
                cohort["scope"],
                cohort["dataset_name"],
                cohort["enrollment_k"],
                row["sample_token"],
            ): row["relevant_rank"]
            for cohort in candidate["cohort_results"]
            if cohort["status"] == "AVAILABLE"
            for row in cohort["query_rows"]
        }
    if set(result) != set(expected):
        raise ValueError("private report lacks B3 or B5-SPATIAL")
    return result


def _dev_outcomes(
    panel: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    b3_vectors: Mapping[str, Any],
    b5_vectors: Mapping[str, Any],
    report_ranks: Mapping[str, Mapping[tuple[str, str, int, str], int]],
) -> tuple[list[QueryOutcome], dict[str, tuple[Sequence[str], Sequence[str]]]]:
    identities = {
        token: row["registered_identity_id"]
        for token, row in records.items()
        if row["registered_identity_id"] is not None
    }
    outcomes: list[QueryOutcome] = []
    population: dict[str, tuple[Sequence[str], Sequence[str]]] = {}
    for cohort in panel["cohorts"]:
        if (
            cohort["scope"] != "DEV"
            or cohort["enrollment_k"] != 1
            or cohort["status"] != "AVAILABLE"
        ):
            continue
        key = (cohort["scope"], cohort["dataset_name"], cohort["enrollment_k"])
        queries, gallery = (
            cohort["query_sample_tokens"],
            cohort["gallery_sample_tokens"],
        )
        population[cohort["dataset_name"]] = (queries, gallery)
        for token in queries:
            b3_ranked = reconstruct_ranking(
                query_token=token,
                gallery_tokens=gallery,
                vectors=b3_vectors,
                identities=identities,
            )
            b5_ranked = reconstruct_ranking(
                query_token=token,
                gallery_tokens=gallery,
                vectors=b5_vectors,
                identities=identities,
            )
            for identifier, ranked in (("B3", b3_ranked), ("B5-SPATIAL", b5_ranked)):
                observed = relevant_rank(ranked)
                if report_ranks[identifier].get((*key, token)) != observed:
                    raise ValueError(
                        f"{identifier} reported relevant rank differs from reconstructed cache ranking"
                    )
            if len(b5_ranked) < 2:
                raise ValueError(
                    "DEV K=1 gallery needs at least two templates for margin selection"
                )
            outcomes.append(
                QueryOutcome(
                    token,
                    key,
                    relevant_rank(b5_ranked),
                    b5_ranked[0].score - b5_ranked[1].score,
                    b3_ranked,
                    b5_ranked,
                )
            )
    return outcomes, population


def _validate_trace_binding(
    trace: Mapping[str, Any],
    inventory: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> None:
    if (
        trace["successor_id"] != "B5-SPATIAL"
        or trace["artifact_bindings"]["evaluation_cache_descriptor_sha256"]
        != descriptor["cache_descriptor_sha256"]
    ):
        raise ValueError("private trace does not bind the B5-SPATIAL evaluation cache")
    records = _records(
        inventory, Path(inventory["inventory"]["artifact_root"]).resolve(strict=True)
    )
    for role, token in trace["private_samples"].items():
        row = records.get(token)
        if row is None:
            raise ValueError("private trace sample is absent from inventory")
        binding = trace["input_bindings"][role.removesuffix("_sample_token")]
        artifact = row["artifact"]
        if any(
            binding[field]
            != artifact[
                {
                    "rgb_sha256": "full_rgb_sha256",
                    "mask_sha256": "full_mask_sha256",
                    "crop_record_sha256": "crop_record_sha256",
                }[field]
            ]
            for field in binding
        ):
            raise ValueError(
                "private trace input binding differs from inventory artifact"
            )


def _selection_basis(name: str) -> str:
    return {
        "high": "rank1 with highest B5 top1-minus-top2 cache-dot margin; token tie-break",
        "middle": "median sorted by relevant rank, negative B5 margin, then token",
        "low": "greatest B5 relevant rank; token tie-break",
    }[name]


if __name__ == "__main__":
    raise SystemExit(main())
