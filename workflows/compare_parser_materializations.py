"""Compare two complete parser materializations by canonical sample token."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.provenance import content_sha256
from embedding.methods.full_segment.preparation.materialization import read_route_plan_bundle
from data.full_segment.route_plan import validate_full128_route_plan_bundle

REPORT_SCHEMA = "cvi.parser_materialization_comparison.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-route-plan", type=Path, required=True)
    parser.add_argument("--baseline-materialization-root", type=Path, required=True)
    parser.add_argument("--candidate-route-plan", type=Path, required=True)
    parser.add_argument("--candidate-materialization-root", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
