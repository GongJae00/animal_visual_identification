"""Compare two complete parser materializations by canonical sample token.

Commands: compare (default), summarize, render-failure, render-comparison.
"""

from __future__ import annotations

import argparse
import sys
import json
from collections import Counter
from pathlib import Path
from typing import Any

import hashlib
import io
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.source_lock import get_record
from shared.foundation.protected_publication import fsync_directory, rename_directory_noreplace
from shared.foundation.retained_file import read_retained_regular_file
from parsing.export.segmentation.full_segment_cache import (
    thaw_animal_parsing_prediction,
    validate_frozen_animal_parsing,
)

from shared.foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from shared.foundation.provenance import content_sha256
from data.full_segment.route_plan import validate_full128_route_plan_bundle


def read_route_plan_bundle(path: Path) -> dict[str, Any]:
    return read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=25_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    ).payload

REPORT_SCHEMA = "cvi.parser_materialization_comparison.v1"


def _run_compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-route-plan", type=Path, required=True)
    parser.add_argument("--baseline-materialization-root", type=Path, required=True)
    parser.add_argument("--candidate-route-plan", type=Path, required=True)
    parser.add_argument("--candidate-materialization-root", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(
        baseline_route_plan=args.baseline_route_plan,
        baseline_materialization_root=args.baseline_materialization_root,
        candidate_route_plan=args.candidate_route_plan,
        candidate_materialization_root=args.candidate_materialization_root,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    write_private_json_bundle(((args.output, report),))
    print(
        json.dumps(
            {
                "status": "CREATED_PARSER_MATERIALIZATION_COMPARISON",
                "output": str(args.output),
                "comparison_sha256": report["comparison_sha256"],
                "sample_count": report["sample_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def compare(
    *,
    baseline_route_plan: Path,
    baseline_materialization_root: Path,
    candidate_route_plan: Path,
    candidate_materialization_root: Path,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    for label in (baseline_label, candidate_label):
        if not label or label != label.strip():
            raise ValueError("comparison labels must be canonical text")
    baseline_bundle = _load_plan(baseline_route_plan)
    candidate_bundle = _load_plan(candidate_route_plan)
    baseline_rows = _rows_by_token(baseline_bundle)
    candidate_rows = _rows_by_token(candidate_bundle)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("parser comparison sample-token populations differ")
    baseline_root = _root(baseline_materialization_root)
    candidate_root = _root(candidate_materialization_root)
    transitions: Counter[tuple[str, str]] = Counter()
    dataset_transitions: Counter[tuple[str, str, str]] = Counter()
    categories: Counter[str] = Counter()
    category_datasets: Counter[tuple[str, str]] = Counter()
    crop_comparisons: Counter[str] = Counter()
    changed_tokens: dict[str, list[str]] = {
        "TERMINAL_TO_BODY_PARSING": [],
        "BODY_PARSING_TO_TERMINAL": [],
        "TERMINAL_REASON_CHANGED": [],
        "BODY_PARSING_CROP_CHANGED": [],
    }
    baseline_receipt_hashes = []
    candidate_receipt_hashes = []
    for token in sorted(baseline_rows):
        baseline_row = baseline_rows[token]
        candidate_row = candidate_rows[token]
        for field in (
            "dataset_name",
            "dataset_version",
            "source_path",
            "source_sha256",
            "source_width",
            "source_height",
            "route_intent",
        ):
            if baseline_row[field] != candidate_row[field]:
                raise ValueError(f"parser comparison {field} differs")
        baseline_receipt, baseline_raw = _receipt(
            baseline_root, baseline_row, baseline_bundle
        )
        candidate_receipt, candidate_raw = _receipt(
            candidate_root, candidate_row, candidate_bundle
        )
        baseline_receipt_hashes.append(baseline_raw)
        candidate_receipt_hashes.append(candidate_raw)
        baseline_state = _state(baseline_receipt)
        candidate_state = _state(candidate_receipt)
        dataset = baseline_row["dataset_name"]
        transitions[(baseline_state, candidate_state)] += 1
        dataset_transitions[(dataset, baseline_state, candidate_state)] += 1
        category = _category(baseline_receipt, candidate_receipt)
        categories[category] += 1
        category_datasets[(category, dataset)] += 1
        if category in changed_tokens:
            changed_tokens[category].append(token)
        if (
            baseline_receipt["actual_route"] == "BODY_PARSING"
            and candidate_receipt["actual_route"] == "BODY_PARSING"
        ):
            same = all(
                baseline_receipt["outputs"][field]
                == candidate_receipt["outputs"][field]
                for field in ("full_rgb_sha256", "full_mask_sha256")
            )
            crop_comparisons["IDENTICAL" if same else "CHANGED"] += 1
            if not same:
                changed_tokens["BODY_PARSING_CROP_CHANGED"].append(token)
    payload = {
        "schema_version": REPORT_SCHEMA,
        "baseline": _binding(
            baseline_bundle,
            baseline_root,
            baseline_label,
            baseline_receipt_hashes,
        ),
        "candidate": _binding(
            candidate_bundle,
            candidate_root,
            candidate_label,
            candidate_receipt_hashes,
        ),
        "sample_count": len(baseline_rows),
        "comparison_scope": (
            "PARSER_POLICY_RUNTIME_AND_ROUTE_SELECTION_COMPARISON;"
            "ROUTE_PLAN_SCHEMAS_MAY_DIFFER"
        ),
        "route_plan_schema_confounded": (
            baseline_bundle["schema_version"] != candidate_bundle["schema_version"]
        ),
        "category_counts": dict(sorted(categories.items())),
        "category_dataset_counts": {
            f"{category}|{dataset}": count
            for (category, dataset), count in sorted(category_datasets.items())
        },
        "transition_counts": {
            f"{left}|{right}": count
            for (left, right), count in sorted(transitions.items())
        },
        "dataset_transition_counts": {
            f"{dataset}|{left}|{right}": count
            for (dataset, left, right), count in sorted(dataset_transitions.items())
        },
        "body_parsing_crop_comparison_counts": dict(sorted(crop_comparisons.items())),
        "changed_sample_tokens": {
            category: tokens for category, tokens in sorted(changed_tokens.items())
        },
        "changed_sample_tokens_sha256": {
            category: content_sha256(tokens)
            for category, tokens in sorted(changed_tokens.items())
        },
    }
    return {**payload, "comparison_sha256": content_sha256(payload)}


def _load_plan(path: Path) -> dict[str, Any]:
    return validate_full128_route_plan_bundle(
        read_route_plan_bundle(path.absolute()), verify_files=False
    )


def _rows_by_token(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["sample_token"]: row for row in bundle["plan"]["records"]}
    if len(rows) != len(bundle["plan"]["records"]):
        raise ValueError("parser comparison route plan repeats sample tokens")
    return rows


def _root(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("parser comparison root must not be a symlink")
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("parser comparison root must be a directory")
    return root


def _receipt(
    root: Path, row: dict[str, Any], bundle: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    document = read_strict_json_document(
        root / "samples" / row["sample_token"] / "execution-receipt.json",
        maximum_bytes=4_194_304,
    )
    receipt = document.payload
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("receipt_sha256") != content_sha256(payload)
        or receipt.get("sample_token") != row["sample_token"]
        or receipt.get("plan_record_sha256") != row["record_sha256"]
        or receipt.get("plan_sha256") != bundle["plan_sha256"]
    ):
        raise ValueError("parser comparison receipt binding differs")
    return receipt, document.raw_sha256


def _state(receipt: dict[str, Any]) -> str:
    if receipt["actual_route"] == "BODY_PARSING":
        return "BODY_PARSING"
    return f"TERMINAL:{receipt['terminal_reason']}"


def _category(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    baseline_body = baseline["actual_route"] == "BODY_PARSING"
    candidate_body = candidate["actual_route"] == "BODY_PARSING"
    if baseline_body and candidate_body:
        return "BODY_PARSING_UNCHANGED"
    if not baseline_body and candidate_body:
        return "TERMINAL_TO_BODY_PARSING"
    if baseline_body and not candidate_body:
        return "BODY_PARSING_TO_TERMINAL"
    if baseline["terminal_reason"] == candidate["terminal_reason"]:
        return "TERMINAL_REASON_UNCHANGED"
    return "TERMINAL_REASON_CHANGED"


def _binding(
    bundle: dict[str, Any], root: Path, label: str, receipt_hashes: list[str]
) -> dict[str, Any]:
    return {
        "label": label,
        "route_plan_bundle_schema": bundle["schema_version"],
        "route_plan_sha256": bundle["plan_sha256"],
        "route_policy_sha256": bundle["route_policy_sha256"],
        "parser_runtime_manifest_sha256": bundle["parser_runtime_manifest_sha256"],
        "parser_policy_sha256": bundle["parser_policy_sha256"],
        "materialization_root": str(root),
        "receipt_file_sha256s_sha256": content_sha256(sorted(receipt_hashes)),
    }


SUMMARY_REPORT_SCHEMA = "cvi.parser_materialization_summary.v1"


def _run_summarize(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report = summarize(
        route_plan=args.route_plan,
        materialization_root=args.materialization_root,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def summarize(
    *, route_plan: Path, materialization_root: Path, allow_partial: bool
) -> dict[str, Any]:
    bundle = validate_full128_route_plan_bundle(
        read_route_plan_bundle(route_plan.absolute()), verify_files=False
    )
    root = materialization_root.resolve(strict=True)
    rows = {row["sample_token"]: row for row in bundle["plan"]["records"]}
    route_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    selection_rule_counts: Counter[str] = Counter()
    reason_dataset_counts: Counter[tuple[str, str]] = Counter()
    missing = []
    receipt_hashes = []
    for token, row in rows.items():
        path = root / "samples" / token / "execution-receipt.json"
        if not path.is_file() or path.is_symlink():
            missing.append(token)
            continue
        document = read_strict_json_document(path, maximum_bytes=4_194_304)
        receipt = document.payload
        payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (
            receipt.get("receipt_sha256") != content_sha256(payload)
            or receipt.get("sample_token") != token
            or receipt.get("plan_record_sha256") != row["record_sha256"]
            or receipt.get("plan_sha256") != bundle["plan_sha256"]
        ):
            raise ValueError("parser materialization receipt binding differs")
        receipt_hashes.append(document.raw_sha256)
        route_counts[receipt["actual_route"]] += 1
        reason = receipt["terminal_reason"]
        if reason is not None:
            reason_counts[reason] += 1
            reason_dataset_counts[(reason, row["dataset_name"])] += 1
        lineage = receipt["parser_lineage"]
        if row["schema_version"] == "cvi.full128_route_plan_record.v3":
            if not isinstance(lineage, dict) or not isinstance(
                lineage.get("selection"), dict
            ):
                raise ValueError("parser v3 receipt lacks selection lineage")
            selection_rule_counts[lineage["selection"]["rule"]] += 1
    if missing and not allow_partial:
        raise ValueError("parser materialization is incomplete")
    return {
        "schema_version": SUMMARY_REPORT_SCHEMA,
        "route_plan_sha256": bundle["plan_sha256"],
        "route_plan_record_count": len(rows),
        "observed_receipt_count": len(receipt_hashes),
        "missing_receipt_count": len(missing),
        "missing_sample_tokens_sha256": content_sha256(sorted(missing)),
        "receipt_file_sha256s_sha256": content_sha256(sorted(receipt_hashes)),
        "actual_route_counts": dict(sorted(route_counts.items())),
        "terminal_reason_counts": dict(sorted(reason_counts.items())),
        "terminal_reason_dataset_counts": {
            f"{reason}|{dataset}": count
            for (reason, dataset), count in sorted(reason_dataset_counts.items())
        },
        "selection_rule_counts": dict(sorted(selection_rule_counts.items())),
    }



_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_REASONS = (
    "NO_PARSED_DOG_INSTANCE",
    "PARSER_INSTANCE_CARDINALITY_AMBIGUOUS",
    "PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS",
    "NO_VALID_PARSED_DOG_INSTANCE",
    "SELECTED_DOG_PARSING_UNUSABLE",
    "AP10K_GLOBAL_BBOX_ASSOCIATION_AMBIGUOUS",
)
_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}
_TILE_SIZE = (300, 230)
_COLUMNS = 4
_ROWS = 6
_PAGE_SIZE = _COLUMNS * _ROWS


def render_parser_failure_review(
    *,
    route_plan: Path,
    materialization_root: Path,
    output_dir: Path,
    samples_per_reason: int = 48,
) -> dict[str, Any]:
    """Render a balanced, token-hash-selected review set without rerunning models."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite parser review: {output_dir}")
    if isinstance(samples_per_reason, bool) or not isinstance(samples_per_reason, int):
        raise TypeError("samples_per_reason must be an integer")
    if samples_per_reason <= 0:
        raise ValueError("samples_per_reason must be positive")
    plan_bundle = validate_full128_route_plan_bundle(
        read_strict_json_document(route_plan, **_LIMITS).payload,
        verify_files=False,
    )
    records = plan_bundle["plan"]["records"]
    if materialization_root.is_symlink():
        raise ValueError("materialization root must not be a symlink")
    root = materialization_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("materialization root must be a regular directory")
    rows_by_token = {row["sample_token"]: row for row in records}
    if len(rows_by_token) != len(records):
        raise ValueError("route plan repeats sample tokens")

    candidates: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    dataset_reason_counts: Counter[tuple[str, str]] = Counter()
    samples_root = root / "samples"
    for sample_dir in sorted(samples_root.iterdir(), key=lambda path: path.name):
        receipt_path = sample_dir / "execution-receipt.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            continue
        receipt = read_strict_json_document(receipt_path, maximum_bytes=262_144).payload
        reason = receipt.get("terminal_reason")
        if reason not in _REASONS:
            continue
        token = receipt.get("sample_token")
        route_row = rows_by_token.get(token)
        if route_row is None or receipt.get("plan_record_sha256") != route_row.get(
            "record_sha256"
        ):
            raise ValueError("parser review receipt and route-plan binding differ")
        candidates[reason].append(route_row)
        reason_counts[reason] += 1
        dataset_reason_counts[(reason, route_row["dataset_name"])] += 1

    selected: dict[str, list[Mapping[str, Any]]] = {}
    observed_reasons = tuple(reason for reason in _REASONS if candidates[reason])
    if not observed_reasons:
        raise ValueError("parser review has no supported failure candidates")
    for reason in observed_reasons:
        available = candidates[reason]
        selected[reason] = _balanced_select(
            available, min(samples_per_reason, len(available)), reason
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        selected_manifest: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        for reason in observed_reasons:
            reason_rows = selected[reason]
            for page_index, start in enumerate(
                range(0, len(reason_rows), _PAGE_SIZE), start=1
            ):
                page_rows = reason_rows[start : start + _PAGE_SIZE]
                rendered = [
                    _load_review_sample(
                        row,
                        root,
                        parser_policy_sha256=plan_bundle["parser_policy_sha256"],
                    )
                    for row in page_rows
                ]
                filename = f"{_reason_slug(reason)}_{page_index:02d}.png"
                image_path = staging / filename
                _render_page(image_path, reason, page_index, rendered)
                image_bytes = image_path.read_bytes()
                pages.append(
                    {
                        "relative_path": filename,
                        "reason": reason,
                        "page_index": page_index,
                        "sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "byte_size": len(image_bytes),
                        "sample_tokens": [
                            item["row"]["sample_token"] for item in rendered
                        ],
                    }
                )
                selected_manifest.extend(
                    _manifest_row(reason, item) for item in rendered
                )
        body = {
            "schema_version": "cvi.parser_failure_visual_review.v1",
            "source_route_plan_sha256": plan_bundle["plan_sha256"],
            "source_route_plan_bundle_sha256": plan_bundle["bundle_sha256"],
            "materialization_root": str(root),
            "selection_method": (
                "BALANCED_BY_DATASET_WITH_UNIQUE_SOURCE_PREFERENCE_THEN_"
                "SHA256_OF_REASON_DATASET_AND_SAMPLE_TOKEN"
            ),
            "samples_per_reason": samples_per_reason,
            "observed_reasons": list(observed_reasons),
            "population_counts_by_reason": dict(sorted(reason_counts.items())),
            "population_counts_by_reason_and_dataset": {
                f"{reason}:{dataset}": count
                for (reason, dataset), count in sorted(dataset_reason_counts.items())
            },
            "selected": selected_manifest,
            "pages": pages,
            "interpretation": (
                "HUMAN_VISUAL_REVIEW_AID_ONLY;DETECTOR_BOXES_AND_PARSER_MASKS_ARE_"
                "MODEL_OUTPUTS_NOT_GROUND_TRUTH"
            ),
        }
        manifest = {**body, "review_sha256": content_sha256(body)}
        write_private_json_bundle(((staging / "review-manifest.json", manifest),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _balanced_select(
    rows: Sequence[Mapping[str, Any]], count: int, reason: str
) -> list[Mapping[str, Any]]:
    by_dataset: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_name"]].append(row)
    for dataset, values in by_dataset.items():
        values.sort(
            key=lambda row: _selection_key(reason, dataset, row["sample_token"])
        )
    selected: list[Mapping[str, Any]] = []
    deferred: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_sources: set[str] = set()
    datasets = sorted(by_dataset)
    while len(selected) < count:
        progressed = False
        for dataset in datasets:
            values = by_dataset[dataset]
            while values:
                candidate = values.pop(0)
                if candidate["source_sha256"] in seen_sources:
                    deferred[dataset].append(candidate)
                    continue
                selected.append(candidate)
                seen_sources.add(candidate["source_sha256"])
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    while len(selected) < count:
        progressed = False
        for dataset in datasets:
            if deferred[dataset]:
                selected.append(deferred[dataset].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def _selection_key(reason: str, dataset: str, token: str) -> str:
    return hashlib.sha256(f"{reason}\0{dataset}\0{token}".encode("ascii")).hexdigest()


def _load_review_sample(
    row: Mapping[str, Any],
    root: Path,
    *,
    parser_policy_sha256: str,
) -> dict[str, Any]:
    dataset_root = Path(get_record(row["dataset_name"]).data_root).resolve(strict=True)
    relative = Path(row["source_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("parser review source path is unsafe")
    source_candidate = dataset_root / relative
    if source_candidate.is_symlink():
        raise ValueError("parser review source must not be a symlink")
    source_path = source_candidate.resolve(strict=True)
    if (
        not source_path.is_relative_to(dataset_root)
        or source_path.is_symlink()
        or not source_path.is_file()
    ):
        raise ValueError("parser review source is unsafe")
    retained = read_retained_regular_file(
        source_path,
        expected_sha256=row["source_sha256"],
        expected_bytes=row["source_byte_size"],
        maximum_bytes=67_108_864,
        capture_payload=True,
        subject="parser review source image",
    )
    assert retained.payload is not None
    with Image.open(io.BytesIO(retained.payload)) as opened:
        opened.load()
        source = opened.convert("RGB")
    if source.size != (row["source_width"], row["source_height"]):
        raise ValueError("parser review source dimensions differ")

    receipt_path = root / "samples" / row["sample_token"] / "execution-receipt.json"
    receipt = read_strict_json_document(receipt_path, maximum_bytes=262_144).payload
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if content_sha256(receipt_body) != receipt.get("receipt_sha256"):
        raise ValueError("parser review execution receipt digest differs")
    parser_key = receipt["parser_lineage"]["parser_cache_key"]
    if parser_key != row["parser_cache_key"]:
        raise ValueError("parser review cache key differs from route plan")
    cache_dir = root / "parser-cache" / parser_key
    cache_receipt = read_strict_json_document(
        cache_dir / "receipt.json", maximum_bytes=1_048_576
    ).payload
    cache_receipt_body = {
        key: value for key, value in cache_receipt.items() if key != "receipt_sha256"
    }
    if (
        content_sha256(cache_receipt_body) != cache_receipt.get("receipt_sha256")
        or cache_receipt.get("receipt_sha256")
        != receipt["parser_lineage"]["parser_cache_receipt_sha256"]
        or cache_receipt.get("parser_cache_key") != parser_key
        or cache_receipt.get("source_sha256") != row["source_sha256"]
        or cache_receipt.get("source_width") != row["source_width"]
        or cache_receipt.get("source_height") != row["source_height"]
        or cache_receipt.get("runtime", {}).get("parser_policy_sha256")
        != parser_policy_sha256
    ):
        raise ValueError("parser review cache receipt binding differs")
    frozen_document = read_strict_json_document(cache_dir / "frozen.json", **_LIMITS)
    frozen = validate_frozen_animal_parsing(frozen_document.payload)
    if (
        frozen_document.raw_sha256 != cache_receipt.get("frozen_json_sha256")
        or frozen["prediction_sha256"] != cache_receipt.get("prediction_sha256")
        or frozen["prediction_sha256"] != receipt["parser_lineage"]["prediction_sha256"]
    ):
        raise ValueError("parser review frozen prediction binding differs")
    prediction = thaw_animal_parsing_prediction(frozen)
    if prediction.policy_sha256 != parser_policy_sha256:
        raise ValueError("parser review policy binding differs")
    if (
        prediction.source_width != source.width
        or prediction.source_height != source.height
    ):
        raise ValueError("parser review prediction dimensions differ")
    return {
        "row": row,
        "source": source,
        "prediction": prediction,
        "receipt": receipt,
    }


def _render_page(
    path: Path, reason: str, page_index: int, samples: Sequence[Mapping[str, Any]]
) -> None:
    card_width = _TILE_SIZE[0] + 24
    card_height = _TILE_SIZE[1] + 106
    width = _COLUMNS * card_width + 24
    height = 112 + _ROWS * card_height + 24
    canvas = Image.new("RGB", (width, height), (10, 16, 28))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(_FONT_BOLD, 25)
    label_font = ImageFont.truetype(_FONT_BOLD, 14)
    small_font = ImageFont.truetype(_FONT, 12)
    draw.text((24, 18), reason, font=title_font, fill=(244, 247, 252))
    draw.text(
        (24, 55),
        f"page {page_index} | red=dog candidate, orange=other candidate, "
        "cyan=mask, green=AP10K dog annotation",
        font=small_font,
        fill=(148, 163, 184),
    )
    draw.text(
        (24, 76),
        "Review labels: TRUE_MISS / WRONG_CLASS / MULTI_DOG_USE_LARGEST / BAD_MASK / DATA_ERROR",
        font=small_font,
        fill=(251, 191, 36),
    )
    for index, item in enumerate(samples):
        row_index, column = divmod(index, _COLUMNS)
        x = 24 + column * card_width
        y = 108 + row_index * card_height
        visual = _annotated_tile(
            item["source"], item["prediction"], item["row"].get("route_evidence")
        )
        canvas.paste(visual, (x, y))
        row = item["row"]
        prediction = item["prediction"]
        dog_instances = [
            instance
            for instance in prediction.instances
            if instance.class_name == "dog"
        ]
        other_count = len(prediction.instances) - len(dog_instances)
        draw.text(
            (x, y + _TILE_SIZE[1] + 7),
            f"{row['dataset_name']} | {row['sample_token'][:12]}",
            font=label_font,
            fill=(226, 232, 240),
        )
        draw.text(
            (x, y + _TILE_SIZE[1] + 29),
            f"candidates {len(prediction.instances)} | dogs {len(dog_instances)} | "
            f"other {other_count}",
            font=small_font,
            fill=(148, 163, 184),
        )
        details = _dog_details(dog_instances)
        draw.text(
            (x, y + _TILE_SIZE[1] + 48),
            details,
            font=small_font,
            fill=(148, 163, 184),
        )
        draw.text(
            (x, y + _TILE_SIZE[1] + 68),
            "review: __________________________",
            font=small_font,
            fill=(251, 191, 36),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=9, optimize=False)


def _annotated_tile(
    source: Image.Image,
    prediction: Any,
    route_evidence: Mapping[str, Any] | None = None,
) -> Image.Image:
    contained = _contain(source, _TILE_SIZE)
    tile, offset, scale = contained
    draw = ImageDraw.Draw(tile)
    annotation_box = (
        route_evidence.get("bbox_xyxy")
        if route_evidence is not None
        and route_evidence.get("kind") == "AP10K_AUTHORITATIVE_BBOX_ASSOCIATION"
        else None
    )
    if (
        isinstance(annotation_box, list)
        and len(annotation_box) == 4
        and all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            for value in annotation_box
        )
    ):
        draw.rectangle(
            _scale_box(annotation_box, offset, scale),
            outline=(74, 222, 128),
            width=3,
        )
    for instance in prediction.instances:
        color = (248, 113, 113) if instance.class_name == "dog" else (251, 146, 60)
        box = _scale_box(instance.detector_box_xyxy, offset, scale)
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0] + 3, box[1] + 3),
            f"{instance.class_name} {instance.class_score:.2f}",
            font=ImageFont.truetype(_FONT_BOLD, 12),
            fill=color,
            stroke_width=2,
            stroke_fill=(10, 16, 28),
        )
        if instance.class_name == "dog":
            mask = Image.fromarray((instance.hard_mask > 0).astype(np.uint8) * 255)
            mask = mask.resize(
                (round(source.width * scale), round(source.height * scale)),
                Image.Resampling.NEAREST,
            )
            mask_canvas = Image.new("L", _TILE_SIZE, 0)
            mask_canvas.paste(mask, offset)
            outline = _mask_outline(np.asarray(mask_canvas, dtype=np.uint8) > 0)
            overlay = np.asarray(tile).copy()
            overlay[outline] = (34, 211, 238)
            tile = Image.fromarray(overlay, mode="RGB")
            draw = ImageDraw.Draw(tile)
    return tile


def _contain(
    image: Image.Image, size: tuple[int, int]
) -> tuple[Image.Image, tuple[int, int], float]:
    scale = min(size[0] / image.width, size[1] / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.BICUBIC)
    offset = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    tile = Image.new("RGB", size, (30, 41, 59))
    tile.paste(resized, offset)
    return tile, offset, scale


def _scale_box(
    box: Sequence[int | float], offset: tuple[int, int], scale: float
) -> tuple[int, int, int, int]:
    return tuple(
        round(value * scale) + offset[index % 2] for index, value in enumerate(box)
    )  # type: ignore[return-value]


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _dog_details(instances: Sequence[Any]) -> str:
    if not instances:
        return "largest dog: none"
    largest = max(instances, key=_box_area)
    return (
        f"largest dog #{largest.instance_index}: score {largest.class_score:.2f} | "
        f"area {_compact_area(_box_area(largest))} | {largest.quality.state}"
    )


def _box_area(instance: Any) -> int:
    return max(0, instance.detector_box_xyxy[2] - instance.detector_box_xyxy[0]) * max(
        0, instance.detector_box_xyxy[3] - instance.detector_box_xyxy[1]
    )


def _compact_area(area: int) -> str:
    if area < 1_000:
        return str(area)
    return f"{area / 1_000:.1f}k"


def _manifest_row(reason: str, item: Mapping[str, Any]) -> dict[str, Any]:
    row = item["row"]
    prediction = item["prediction"]
    instances = [
        {
            "instance_index": instance.instance_index,
            "class_name": instance.class_name,
            "class_score": instance.class_score,
            "detector_box_xyxy": list(instance.detector_box_xyxy),
            "mask_box_xyxy": (
                None if instance.mask_box_xyxy is None else list(instance.mask_box_xyxy)
            ),
            "quality_state": instance.quality.state,
            "quality_reasons": list(instance.quality.reasons),
            "foreground_pixels": instance.quality.foreground_pixels,
        }
        for instance in prediction.instances
    ]
    payload = {
        "reason": reason,
        "sample_token": row["sample_token"],
        "dataset_name": row["dataset_name"],
        "source_path": row["source_path"],
        "source_sha256": row["source_sha256"],
        "source_width": row["source_width"],
        "source_height": row["source_height"],
        "parser_prediction_sha256": item["receipt"]["parser_lineage"][
            "prediction_sha256"
        ],
        "instances": instances,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _reason_slug(reason: str) -> str:
    return {
        "NO_PARSED_DOG_INSTANCE": "01_no_parsed_dog",
        "PARSER_INSTANCE_CARDINALITY_AMBIGUOUS": "02_multiple_instances",
        "PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS": "02_distinct_dogs",
        "NO_VALID_PARSED_DOG_INSTANCE": "03_no_valid_dog",
        "SELECTED_DOG_PARSING_UNUSABLE": "04_unusable_mask",
        "AP10K_GLOBAL_BBOX_ASSOCIATION_AMBIGUOUS": "05_ap10k_bbox_ambiguous",
    }[reason]


def _run_render_failure(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-plan", required=True, type=Path)
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-reason", type=int, default=48)
    args = parser.parse_args(argv)
    manifest = render_parser_failure_review(
        route_plan=args.route_plan,
        materialization_root=args.materialization_root,
        output_dir=args.output_dir,
        samples_per_reason=args.samples_per_reason,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_PARSER_FAILURE_VISUAL_REVIEW",
                "review_sha256": manifest["review_sha256"],
                "selected_sample_count": len(manifest["selected"]),
                "page_count": len(manifest["pages"]),
                "output": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0



_CATEGORIES = ("TERMINAL_TO_BODY_PARSING", "BODY_PARSING_CROP_CHANGED")
_PAIR_SIZE = (300, 230)
_COLUMNS = 2
_ROWS = 4
_PAGE_SIZE = _COLUMNS * _ROWS


def _run_render_comparison(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline-route-plan", type=Path, required=True)
    parser.add_argument("--candidate-route-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=24)
    args = parser.parse_args(argv)
    manifest = render(
        comparison_path=args.comparison,
        baseline_route_plan=args.baseline_route_plan,
        candidate_route_plan=args.candidate_route_plan,
        output_dir=args.output_dir,
        samples_per_category=args.samples_per_category,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_PARSER_MATERIALIZATION_COMPARISON_REVIEW",
                "output": str(args.output_dir),
                "review_sha256": manifest["review_sha256"],
                "page_count": len(manifest["pages"]),
                "selected_sample_count": len(manifest["selected"]),
            },
            sort_keys=True,
        )
    )
    return 0


def render(
    *,
    comparison_path: Path,
    baseline_route_plan: Path,
    candidate_route_plan: Path,
    output_dir: Path,
    samples_per_category: int,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if (
        isinstance(samples_per_category, bool)
        or not isinstance(samples_per_category, int)
        or samples_per_category <= 0
    ):
        raise ValueError("samples_per_category must be positive")
    comparison_document = read_strict_json_document(
        comparison_path, maximum_bytes=134_217_728
    )
    comparison = comparison_document.payload
    comparison_payload = {
        key: value for key, value in comparison.items() if key != "comparison_sha256"
    }
    if (
        comparison.get("schema_version") != "cvi.parser_materialization_comparison.v1"
        or comparison.get("comparison_sha256") != content_sha256(comparison_payload)
    ):
        raise ValueError("parser comparison report differs")
    baseline_bundle = _plan(baseline_route_plan)
    candidate_bundle = _plan(candidate_route_plan)
    _validate_binding(comparison["baseline"], baseline_bundle)
    _validate_binding(comparison["candidate"], candidate_bundle)
    baseline_rows = {row["sample_token"]: row for row in baseline_bundle["plan"]["records"]}
    candidate_rows = {row["sample_token"]: row for row in candidate_bundle["plan"]["records"]}
    baseline_root = Path(comparison["baseline"]["materialization_root"]).resolve(strict=True)
    candidate_root = Path(comparison["candidate"]["materialization_root"]).resolve(strict=True)
    selected = {}
    for category in _CATEGORIES:
        tokens = comparison["changed_sample_tokens"][category]
        selected[category] = _balanced_tokens(
            tokens,
            baseline_rows,
            min(samples_per_category, len(tokens)),
            category,
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        pages = []
        manifest_rows = []
        for category in _CATEGORIES:
            tokens = selected[category]
            for page_index, start in enumerate(range(0, len(tokens), _PAGE_SIZE), start=1):
                page_tokens = tokens[start : start + _PAGE_SIZE]
                pairs = []
                for token in page_tokens:
                    baseline = _load_review_sample(
                        baseline_rows[token],
                        baseline_root,
                        parser_policy_sha256=baseline_bundle["parser_policy_sha256"],
                    )
                    candidate = _load_review_sample(
                        candidate_rows[token],
                        candidate_root,
                        parser_policy_sha256=candidate_bundle["parser_policy_sha256"],
                    )
                    pairs.append((baseline, candidate))
                    manifest_rows.append(_comparison_manifest_row(category, baseline, candidate))
                filename = f"{category.lower()}_{page_index:02d}.png"
                path = staging / filename
                _comparison_render_page(path, category, page_index, pairs)
                payload = path.read_bytes()
                pages.append(
                    {
                        "relative_path": filename,
                        "category": category,
                        "page_index": page_index,
                        "sample_tokens": list(page_tokens),
                        "byte_size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        body = {
            "schema_version": "cvi.parser_materialization_comparison_review.v1",
            "comparison_sha256": comparison["comparison_sha256"],
            "baseline_route_plan_sha256": baseline_bundle["plan_sha256"],
            "candidate_route_plan_sha256": candidate_bundle["plan_sha256"],
            "selection_method": (
                "BALANCED_BY_DATASET_THEN_SHA256_OF_CATEGORY_DATASET_AND_SAMPLE_TOKEN"
            ),
            "samples_per_category": samples_per_category,
            "selected": manifest_rows,
            "pages": pages,
            "interpretation": (
                "PAIRED_MODEL_OUTPUT_REVIEW_ONLY;LEFT_BASELINE_RIGHT_CANDIDATE;"
                "NOT_GROUND_TRUTH"
            ),
        }
        manifest = {**body, "review_sha256": content_sha256(body)}
        write_private_json_bundle(((staging / "review-manifest.json", manifest),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _plan(path: Path) -> dict[str, Any]:
    return validate_full128_route_plan_bundle(
        read_route_plan_bundle(path.absolute()), verify_files=False
    )


def _validate_binding(binding: object, bundle: dict[str, Any]) -> None:
    if not isinstance(binding, dict) or any(
        binding.get(binding_field) != bundle[bundle_field]
        for binding_field, bundle_field in (
            ("route_plan_sha256", "plan_sha256"),
            ("route_policy_sha256", "route_policy_sha256"),
            ("parser_runtime_manifest_sha256", "parser_runtime_manifest_sha256"),
            ("parser_policy_sha256", "parser_policy_sha256"),
        )
    ):
        raise ValueError("parser comparison route-plan binding differs")


def _balanced_tokens(
    tokens: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    count: int,
    category: str,
) -> list[str]:
    by_dataset: defaultdict[str, list[str]] = defaultdict(list)
    for token in tokens:
        by_dataset[rows[token]["dataset_name"]].append(token)
    for dataset, values in by_dataset.items():
        values.sort(
            key=lambda token: hashlib.sha256(
                f"{category}\0{dataset}\0{token}".encode("ascii")
            ).hexdigest()
        )
    selected = []
    while len(selected) < count:
        progressed = False
        for dataset in sorted(by_dataset):
            if by_dataset[dataset]:
                selected.append(by_dataset[dataset].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def _comparison_render_page(
    path: Path,
    category: str,
    page_index: int,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    pair_width = _PAIR_SIZE[0] * 2 + 28
    pair_height = _PAIR_SIZE[1] + 92
    canvas = Image.new(
        "RGB",
        (24 + _COLUMNS * pair_width, 100 + _ROWS * pair_height),
        (10, 16, 28),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(_FONT_BOLD, 24)
    label_font = ImageFont.truetype(_FONT_BOLD, 13)
    small_font = ImageFont.truetype(_FONT, 12)
    draw.text((24, 16), category, font=title_font, fill=(244, 247, 252))
    draw.text(
        (24, 52),
        f"page {page_index} | left=v5 route-v2 | right=v6 route-v3",
        font=small_font,
        fill=(148, 163, 184),
    )
    for index, (baseline, candidate) in enumerate(pairs):
        row_index, column = divmod(index, _COLUMNS)
        x = 24 + column * pair_width
        y = 92 + row_index * pair_height
        left = _annotated_tile(
            baseline["source"], baseline["prediction"], baseline["row"]["route_evidence"]
        )
        right = _annotated_tile(
            candidate["source"], candidate["prediction"], candidate["row"]["route_evidence"]
        )
        canvas.paste(left, (x, y))
        canvas.paste(right, (x + _PAIR_SIZE[0] + 8, y))
        token = baseline["row"]["sample_token"]
        dataset = baseline["row"]["dataset_name"]
        draw.text(
            (x, y + _PAIR_SIZE[1] + 7),
            f"{dataset} | {token[:12]}",
            font=label_font,
            fill=(226, 232, 240),
        )
        draw.text(
            (x, y + _PAIR_SIZE[1] + 28),
            f"v5 {_receipt_state(baseline['receipt'])}",
            font=small_font,
            fill=(248, 113, 113),
        )
        draw.text(
            (x, y + _PAIR_SIZE[1] + 47),
            f"v6 {_receipt_state(candidate['receipt'])}",
            font=small_font,
            fill=(74, 222, 128),
        )
    canvas.save(path, format="PNG", compress_level=9, optimize=False)


def _receipt_state(receipt: Mapping[str, Any]) -> str:
    return (
        "BODY_PARSING"
        if receipt["actual_route"] == "BODY_PARSING"
        else str(receipt["terminal_reason"])
    )


def _comparison_manifest_row(
    category: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "category": category,
        "sample_token": baseline["row"]["sample_token"],
        "dataset_name": baseline["row"]["dataset_name"],
        "source_sha256": baseline["row"]["source_sha256"],
        "baseline_state": _receipt_state(baseline["receipt"]),
        "candidate_state": _receipt_state(candidate["receipt"]),
        "baseline_prediction_sha256": baseline["receipt"]["parser_lineage"][
            "prediction_sha256"
        ],
        "candidate_prediction_sha256": candidate["receipt"]["parser_lineage"][
            "prediction_sha256"
        ],
    }
    return {**payload, "record_sha256": content_sha256(payload)}




def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "compare"
    if argv and argv[0] in {"compare", "summarize", "render-failure", "render-comparison"}:
        command = argv[0]
        argv = argv[1:]
    return {
        "compare": _run_compare,
        "summarize": _run_summarize,
        "render-failure": _run_render_failure,
        "render-comparison": _run_render_comparison,
    }[command](argv)


if __name__ == "__main__":
    raise SystemExit(main())
